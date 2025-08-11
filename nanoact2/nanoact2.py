import os
import re
import time
import json
import gzip
import shutil
import tarfile
import logging
import urllib.parse
from subprocess import Popen, PIPE, run
from typing import Optional, Iterator, Dict, Any, List, Tuple, TextIO, Union
import requests
import zipfile
import platform 
import hdbscan
import numpy as np
import edlib
import xmltodict
import pandas as pd
from requests import get, post
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from collections import Counter
from random import sample as random_sample

# 1. 先清除預設的 handler（避免重複設定多次會有累積行為）
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

# 2. 設定 logging 的最小顯示等級（DEBUG、INFO、WARNING、ERROR、CRITICAL）
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)

# 3. 取得 logger 實例（可以用 __name__ 讓你知道是哪個模組在印）
logger = logging.getLogger(__name__)

# 4. 測試輸出：
logger.debug("這是一條 DEBUG 訊息，只有等級>=DEBUG 才會顯示")
logger.info("這是一條 INFO 訊息，等級>=INFO 才會顯示")
logger.warning("這是一條 WARNING 訊息")
logger.error("這是一條 ERROR 訊息")

class NanoAct:
    def __init__(self, TEMP: str = "./temp/", mafft_platform: Optional[str] = None) -> None:
        """
        Initialize NanoAct with a temporary directory.
        Optionally, specify mafft_platform as "windows", "linux", or None to auto-detect.
        """
        self.TEMP = TEMP
        os.makedirs(self.TEMP, exist_ok=True)

        if TEMP == "./temp/":
            logger.info(f"Temp folder is set to {self.TEMP}. "
                        f"You can change it by NanoAct(TEMP='your_temp_folder').")
            logger.info("We recommend not setting the temp folder to NFS, FTP, Samba, Google Drive, etc., "
                        "as it may cause unexpected errors.")

        self.fasta_ext = ["fasta", "fa", "fas"]
        self.fastq_ext = ["fastq"]
        self.lib_path = os.path.dirname(os.path.realpath(__file__))
        self.tax_id_cache = os.path.join(self.lib_path, "taxid_cache", "taxid_cache.json")

        # mafft_platform: "windows", "linux", or None (auto-detect)
        self.mafft_platform = mafft_platform

    def _lib_path(self) -> str:
        """
        Get the filesystem path of this library.
        """
        return self.lib_path

    def _exec(self, cmd: str, suppress_output: bool = True) -> None:
        """
        Execute a shell command. If suppress_output is True, suppress stdout/stderr.
        """
        if suppress_output:
            with open(os.devnull, "w") as DEVNULL:
                run(cmd, stdout=DEVNULL, stderr=DEVNULL, shell=True)
        else:
            result = run(cmd, stdout=PIPE, stderr=PIPE, shell=True)
            stdout_str = result.stdout.decode("utf-8", errors="ignore")
            stderr_str = result.stderr.decode("utf-8", errors="ignore")
            logger.info(f">> {cmd}")
            logger.info(f"Output:\n{stdout_str}")
            if stderr_str:
                logger.error(f"Error:\n{stderr_str}")

    def _clean_temp(self) -> None:
        """
        Remove and recreate the temp directory.
        """
        try:
            shutil.rmtree(self.TEMP)
        except FileNotFoundError:
            pass
        os.makedirs(self.TEMP, exist_ok=True)

    def _exec_rt(self, cmd: str, prefix: str = "") -> None:
        """
        Execute a shell command and stream its stdout in real time.
        """
        process = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        assert process.stdout is not None  # for type checker
        for line in iter(process.stdout.readline, b""):
            logger.info(f"{prefix}>>> {line.rstrip().decode('utf-8', errors='ignore')}")

    def _IUPACde(self, seq: str) -> str:
        """
        Convert IUPAC ambiguous DNA codes in a sequence to regex character classes.
        """
        seq = seq.upper()
        replacements = {
            "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
            "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
            "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]"
        }
        for code, repl in replacements.items():
            seq = seq.replace(code, repl)
        return seq

    def _extract_degenerate_seq(self, seq: str) -> List[str]:
        """
        Expand an IUPAC-degenerate sequence into all possible non-degenerate sequences.
        Requires self.sre_yield.AllStrings to be defined elsewhere.
        """
        pattern = self._IUPACde(seq)
        return list(self.sre_yield.AllStrings(pattern))

    def _fastq_reader(self, handle: TextIO, suppress_warning: bool = True) -> Iterator[Dict[str, str]]:
        """
        Custom FASTQ reader that yields dictionaries with keys: title, seq, qual.
        If quality length mismatches sequence length, it pads quality with '#'.
        """
        line_counter = 0
        while True:
            line = handle.readline()
            if not line:
                return
            line_counter += 1
            if not line.startswith("@"):
                if not suppress_warning:
                    logger.warning(f"Skipped line {line_counter}: does not start with '@'. FASTQ file may be corrupted.")
                continue

            title = line[1:].rstrip()
            seq = handle.readline().rstrip()
            handle.readline()  # skip the '+' line
            qual = handle.readline().rstrip()
            if len(qual) != len(seq):
                if not suppress_warning:
                    logger.warning(f"Inconsistency found in record '{title}': quality length != sequence length.")
                diff = len(seq) - len(qual)
                yield {"title": title, "seq": seq, "qual": qual + "#" * diff}
            else:
                yield {"title": title, "seq": seq, "qual": qual}

    def _fastq_writer(self, title: str, seq: str, qual: str, handle: TextIO) -> None:
        """
        Write a single FASTQ record to the given file handle.
        """
        handle.write(f"@{title}\n{seq}\n+\n{qual}\n")

    def _fasta_reader(self, handle: TextIO) -> Iterator[Dict[str, str]]:
        """
        Custom FASTA reader that yields dictionaries with keys: title, seq.
        """
        header_line = handle.readline()
        while header_line:
            if header_line.startswith(">"):
                title = header_line[1:].rstrip()
                seq_parts = []
                while True:
                    pos = handle.tell()
                    line = handle.readline()
                    if not line or line.startswith(">"):
                        handle.seek(pos)
                        break
                    seq_parts.append(line.rstrip())
                yield {"title": title, "seq": "".join(seq_parts).replace(" ", "")}
            header_line = handle.readline()

    def _count_seq_num(self, fastq: str) -> int:
        """
        Count the number of records in a FASTQ file (lines/4).
        """
        with open(fastq, "r") as fh:
            total_lines = sum(1 for _ in fh)
        return total_lines // 4

    def _fastq_rename_title(self, src: str, des: str) -> None:
        """
        Rename all FASTQ titles in 'src' file to incremental IDs and write to 'des'.
        """
        with open(src, "r") as f_in, open(des, "w") as f_out:
            counter = 0
            for line in f_in:
                if line.startswith("@"):
                    f_out.write(f"@seq{counter}\n")
                    counter += 1
                else:
                    f_out.write(line)

    def _pairwise_distance(self, s1: str, s2: str) -> float:
        """
        Compute a normalized edit-distance-based pairwise distance (%) between two sequences.
        """
        s1_upper = s1.upper()
        s2_upper = s2.upper()
        ed = edlib.align(s1_upper, s2_upper)["editDistance"]
        return ed / (min(len(s1), len(s2)) + 0.1) * 100

    def _reverse_complement(self, s: str) -> str:
        """
        Return the reverse complement of a DNA sequence.
        """
        complement_map = {
            "A": "T", "T": "A", "C": "G", "G": "C",
            "a": "t", "t": "a", "c": "g", "g": "c",
            "Y": "R", "R": "Y", "S": "S", "W": "W",
            "y": "r", "r": "y", "s": "s", "w": "w",
            "K": "M", "M": "K", "B": "V", "V": "B",
            "N": "N", "n": "n"
        }
        return "".join(complement_map.get(base, base) for base in reversed(s))

    def _check_input_output(self, input_format: str, output_format: str) -> Dict[str, Any]:
        """
        Verify that input_format and output_format are valid.
        Returns a dict specifying how to handle conversions.
        """
        valid_inputs = self.fasta_ext + self.fastq_ext + ["both"]
        valid_outputs = self.fasta_ext + self.fastq_ext + ["both"]

        if input_format not in valid_inputs:
            raise ValueError(f"Input format must be one of {valid_inputs}")
        if output_format not in valid_outputs:
            raise ValueError(f"Output format must be one of {valid_outputs}")

        if input_format in self.fasta_ext and output_format in ["fastq", "both"]:
            raise ValueError("FASTA does not contain quality scores and cannot be converted to FASTQ.")

        result: Dict[str, Any] = {"input": [], "output": {}}
        # input: if 'both', allow all fasta_ext + fastq_ext; otherwise single extension
        if input_format == "both":
            result["input"] = self.fasta_ext + self.fastq_ext
        else:
            result["input"] = [input_format]

        # output: if 'both', set both; otherwise single
        if output_format in self.fasta_ext:
            result["output"]["fasta"] = output_format
        elif output_format in self.fastq_ext:
            result["output"]["fastq"] = output_format
        elif output_format == "both":
            result["output"]["fasta"] = "fas"
            result["output"]["fastq"] = "fastq"
        return result  
    def NCBIblast(self, seqs: str = ">a\nTTGTCTCCAAGATTAAGCCATGCATGTCTAAGTATAAGCAATTATACCGCGGGGGCACGAATGGCTCATTATATAAGTTATCGTTTATTTGATAGCACATTACTACATGGATAACTGTGG\n>b\nTAATACATGCTAAAAATCCCGACTTCGGAAGGGATGTATTTATTGGGTCGCTTAACGCCCTTCAGGCTTCCTGGTGATT\n",
                  timeout: int = 30) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Submit nucleotide sequences to NCBI BLAST (MEGABLAST) and retrieve the top hit for each query.
        Returns a dictionary mapping query IDs to hit details, or None on timeout/failure.
        """
        program = "blastn&MEGABLAST=on"
        database = "core_nt"
        encoded_queries = urllib.parse.quote(seqs)
        word_size = 32
        expect = 0.001

        # Build the POST data
        args = (
            f"CMD=Put&PROGRAM={program}&DATABASE={database}"
            f"&QUERY={encoded_queries}&WORD_SIZE={word_size}&EXPECT={expect}"
        )
        url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"

        try:
            response = post(url, data=args)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to submit BLAST request: {e}")
            return None

        # Parse out the Request ID (RID) from the response
        rid = ""
        for line in response.text.splitlines():
            if line.startswith("    RID = "):
                parts = line.split()
                if len(parts) >= 3:
                    rid = parts[2]
                    break

        if not rid:
            logger.error("Could not retrieve RID from BLAST submission response.")
            return None

        logger.info(f"Query {rid} submitted.")
        logger.info(f"Check status: https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_OBJECT=SearchInfo&RID={rid}")
        logger.info(f"View results: https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&RID={rid}")

        # Poll for results
        retries = timeout * 2  # check every 30 seconds
        status_url = f"https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_OBJECT=SearchInfo&RID={rid}"

        while retries > 0:
            time.sleep(30)
            retries -= 1
            try:
                status_resp = get(status_url)
                status_resp.raise_for_status()
            except Exception as e:
                logger.warning(f"Failed to check BLAST status for RID {rid}: {e}")
                continue

            resp_text = status_resp.text
            if "Status=WAITING" in resp_text:
                logger.info("BLAST search is still running...")
                continue
            if "Status=FAILED" in resp_text:
                logger.error(f"BLAST search {rid} failed; please contact blast-help@ncbi.nlm.nih.gov.")
                return None
            if "Status=UNKNOWN" in resp_text:
                logger.error(f"BLAST search {rid} expired or unknown.")
                return None
            if "Status=READY" in resp_text:
                if "ThereAreHits=yes" in resp_text:
                    logger.info("BLAST search complete, retrieving results...")
                else:
                    logger.info("BLAST search complete, but no hits found.")
                break

        if retries == 0:
            logger.error(f"Search {rid} timed out after {timeout} minutes.")
            return None

        # Retrieve results in XML format
        result_url = f"https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&RID={rid}&FORMAT_TYPE=XML"
        logger.info(f"Retrieving results from {result_url}")
        try:
            result_resp = get(result_url)
            result_resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to retrieve BLAST results for RID {rid}: {e}")
            return None

        # Parse XML to dict
        try:
            blast_dict = xmltodict.parse(result_resp.text)
        except Exception as e:
            logger.error(f"Failed to parse BLAST XML response: {e}")
            return None

        pool: Dict[str, Dict[str, Any]] = {}
        iterations = blast_dict.get("BlastOutput", {}) \
                               .get("BlastOutput_iterations", {}) \
                               .get("Iteration", [])
        # Normalize to list if only one iteration
        if isinstance(iterations, dict):
            iterations = [iterations]

        for rec in iterations:
            seq_name = rec.get("Iteration_query-def", "")
            if not seq_name:
                logger.warning("Missing query definition in one BLAST iteration.")
                continue

            hits = rec.get("Iteration_hits", {}).get("Hit")
            top_hit = None
            if isinstance(hits, list) and hits:
                top_hit = hits[0]
            elif isinstance(hits, dict):
                top_hit = hits

            if not top_hit:
                pool[seq_name] = {"acc": None, "hit_seq": None, "hit_def": None,
                                  "similarity": 0.0, "org": None}
                continue

            acc = top_hit.get("Hit_accession", "")
            hsps = top_hit.get("Hit_hsps", {}).get("Hsp", [])
            if isinstance(hsps, list) and hsps:
                hit_hsp = hsps[0]
            elif isinstance(hsps, dict):
                hit_hsp = hsps
            else:
                logger.warning(f"No HSP data for hit {acc} of query {seq_name}.")
                continue

            hit_seq = hit_hsp.get("Hsp_hseq", "").replace("-", "")
            hit_def = top_hit.get("Hit_def", "")
            identity = int(hit_hsp.get("Hsp_identity", 0))
            align_len = int(hit_hsp.get("Hsp_align-len", 1))
            similarity = round(identity / align_len, 2)

            pool[seq_name] = {
                "acc": acc,
                "hit_seq": hit_seq,
                "hit_def": hit_def,
                "similarity": similarity,
                "org": "",
            }

            # Fetch taxonomic info if accession exists
            if acc:
                try:
                    taxon_info_uri = (
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                        f"db=nuccore&rettype=gb&retmode=xml&id={acc}"
                    )
                    tax_resp = get(taxon_info_uri)
                    tax_resp.raise_for_status()
                    tax_dict = xmltodict.parse(tax_resp.text)
                    gbseq = tax_dict.get("GBSet", {}).get("GBSeq", {})
                    org = gbseq.get("GBSeq_organism", "")
                    taxid = ""
                    for feature in gbseq.get("GBSeq_feature-table", {}).get("GBFeature", []):
                        if feature.get("GBFeature_key") == "source":
                            for qualifier in feature.get("GBFeature_quals", {}).get("GBQualifier", []):
                                if qualifier.get("GBQualifier_name") == "db_xref":
                                    val = qualifier.get("GBQualifier_value", "")
                                    if val.startswith("taxon:"):
                                        taxid = val.split(":")[-1]
                                        break
                            if taxid:
                                break

                    pool[seq_name].update({"org": org, "taxid": taxid})
                except Exception as e:
                    logger.warning(f"Failed to fetch taxon info for accession {acc}: {e}")

                # Fetch lineage ranks if taxid found
                ranks: Dict[str, str] = {
                    "kingdom": "incertae sedis", "phylum": "incertae sedis",
                    "class": "incertae sedis", "order": "incertae sedis",
                    "family": "incertae sedis", "genus": "incertae sedis"
                }
                if pool[seq_name].get("taxid"):
                    try:
                        taxid_info_uri = (
                            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                            f"db=taxonomy&rettype=xml&id={pool[seq_name]['taxid']}"
                        )
                        taxid_resp = get(taxid_info_uri)
                        taxid_resp.raise_for_status()
                        taxid_dict = xmltodict.parse(taxid_resp.text)
                        lineage = taxid_dict.get("TaxaSet", {}).get("Taxon", {}) \
                                              .get("LineageEx", {}).get("Taxon", [])
                        for entry in lineage:
                            rank = entry.get("Rank", "")
                            sci_name = entry.get("ScientificName", "")
                            if rank in ranks:
                                ranks[rank] = sci_name
                        pool[seq_name].update(ranks)
                    except Exception as e:
                        logger.warning(f"Failed to fetch lineage for taxid {pool[seq_name]['taxid']}: {e}")

        return pool

    def _funguild(self, org: str = "Fusarium") -> Tuple[Optional[str], Optional[str]]:
        """
        Query the Funguild database for a given organism name.
        Returns (guild, notes) or (None, None) if not found.
        """
        guild: Optional[str] = None
        notes: Optional[str] = None

        try:
            url = (
                "https://www.mycoportal.org/funguild/services/api/"
                f"db_return.php?qDB=funguild_db&qField=taxon&qText={org}"
            )
            response = get(url)
            response.raise_for_status()
            data = json.loads(response.text)
        except Exception as e:
            logger.warning(f"Failed to load Funguild data for {org}: {e}")
            return None, None

        if isinstance(data, list) and data:
            entry = data[0]
            guild = entry.get("guild")
            notes = entry.get("notes")

        return guild, notes

    def blast_2(
        self,
        src: str,
        des: str,
        name: str = "blast.csv",
        funguild: bool = True,
        startswith: str = "con_",
        input_format: str = "fasta",
        query_range: Tuple[Optional[int], Optional[int]] = (None, None),
        batch: int = 5,
        timeout: int = 30,
    ) -> Optional[str]:
        """
        Search all sequence files in 'src' directory (matching prefix) via NCBI blast.
        Consolidate results into a DataFrame, optionally augment with FuGuild data, and save CSV to 'des/name'.
        Returns the CSV path or None if no sequences.
        """
        # Collect all sequences into a DataFrame
        pool_df = pd.DataFrame()
        query_seqs: List[str] = []
        in_exts = self._check_input_output(input_format, "fasta")["input"]

        for entry in os.scandir(src):
            if not entry.name.startswith(startswith):
                continue
            filename, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            if entry.is_file() and ext in in_exts:
                with open(entry.path, "r") as handle:
                    if ext in self.fasta_ext:
                        seqs = list(self._fasta_reader(handle))
                    elif ext in self.fastq_ext:
                        seqs = list(self._fastq_reader(handle))
                    else:
                        continue

                    for s in seqs:
                        pool_df = pd.concat([pool_df, pd.DataFrame([s])], ignore_index=True)
                        seq_str = s["seq"]
                        start, end = query_range
                        q = seq_str[start:end] if start is not None or end is not None else seq_str
                        if not q:
                            logger.warning(f"Zero query length for '{s['title']}'. Check query_range. Skipping.")
                            continue
                        query_seqs.append(f">{s['title']}\n{q}")

        if not query_seqs:
            logger.info("No sequence found, exiting blast_2.")
            return None

        pool_df.set_index("title", inplace=True)

        # Annotate pool_df with fasta string, length, and parse sample information from title
        for idx, row in pool_df.iterrows():
            seq = row.get("seq", "")
            pool_df.at[idx, "fasta"] = f">{idx}\n{seq}\n"
            pool_df.at[idx, "length"] = len(seq)

            try:
                sample, cluster_no, reads_count = re.search(
                    r"(.*)_cluster_([-0-9]+)_r(\d+)", idx
                ).groups()
                pool_df.at[idx, "sample"] = sample
                pool_df.at[idx, "cluster_no"] = int(cluster_no)
                pool_df.at[idx, "reads_count"] = int(reads_count)
            except Exception:
                logger.debug(f"Title '{idx}' did not match clustering pattern.")

        # Perform BLAST in batches
        blast_result_pool: Dict[str, Dict[str, Any]] = {}
        total = len(query_seqs)
        i = 0
        while i < total:
            end_i = min(i + batch, total)
            logger.info(f"Blasting queries {i + 1} to {end_i} of {total}.")
            query_block = "\n".join(query_seqs[i:end_i])

            retry_count = 3
            while retry_count > 0:
                try:
                    result = self.NCBIblast(query_block, timeout=timeout)
                    if result:
                        blast_result_pool.update(result)
                        break
                    retry_count -= 1
                    logger.warning(f"Retrying BLAST for queries {i + 1}-{end_i}. Attempts left: {retry_count}.")
                except Exception as e:
                    retry_count -= 1
                    logger.warning(f"Error during BLAST for queries {i + 1}-{end_i}: {e}. Retries left: {retry_count}.")
            i += batch

        # Merge BLAST results into DataFrame
        for sample_id, hit_info in blast_result_pool.items():
            for key, value in hit_info.items():
                pool_df.at[sample_id, key] = value

            # Optionally augment with FuGuild data
            if funguild and hit_info.get("org"):
                guild, notes = self._funguild(hit_info["org"])
                if guild is not None:
                    pool_df.at[sample_id, "funguild"] = guild
                if notes is not None:
                    pool_df.at[sample_id, "funguild_notes"] = notes

        # Ensure output directory exists
        os.makedirs(des, exist_ok=True)
        output_path = os.path.join(des, name)
        try:
            pool_df.to_csv(output_path, encoding="utf-8-sig")
            logger.info(f"BLAST results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write BLAST results CSV: {e}")
            return None

        return output_path
    def _ensure_mafft_installed(self) -> str:
        """
        Ensure MAFFT is installed under lib_path/bin. Download and extract
        the appropriate package if missing. Returns the path to the mafft.bat binary.
        """
        bin_dir = os.path.join(self.lib_path, "bin")
        os.makedirs(bin_dir, exist_ok=True)

        # Determine target platform
        sys_plat = self.mafft_platform or platform.system().lower()
        if sys_plat.startswith("win"):
            plat_key = "windows"
            mafft_folder = os.path.join(bin_dir, "mafft-win")
            download_url = "https://mafft.cbrc.jp/alignment/software/mafft-7.526-win64-signed.zip"
            archive_name = os.path.join(self.TEMP, "mafft-win64.zip")
        else:
            plat_key = "linux"
            mafft_folder = os.path.join(bin_dir, "mafft-linux64")
            download_url = "https://mafft.cbrc.jp/alignment/software/mafft-7.526-linux.tgz"
            archive_name = os.path.join(self.TEMP, "mafft-linux.tgz")

        mafft_binary = os.path.join(mafft_folder, "mafft.bat")

        if not os.path.isfile(mafft_binary):
            logger.info(f"MAFFT not found for '{plat_key}'. Downloading from {download_url}")
            # Download the archive
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            with open(archive_name, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)

            # Extract archive
            logger.info(f"Extracting MAFFT to {bin_dir}")
            if plat_key == "windows":
                with zipfile.ZipFile(archive_name, "r") as zip_ref:
                    zip_ref.extractall(bin_dir)
                # The zip unpacks into a folder like "mafft-win64" automatically
            else:
                with tarfile.open(archive_name, "r:gz") as tar_ref:
                    tar_ref.extractall(bin_dir)
                # The tgz unpacks into "mafft-linux64"
            logger.info("MAFFT installation complete.")

        return mafft_binary

    def _mafft(self, src: str, des: str, adjustdirection: bool = False) -> None:
        """
        Run MAFFT to align sequences in 'src' and write output to 'des'.
        If adjustdirection is True, add the --adjustdirection flag.
        Automatically downloads and uses the correct MAFFT binary for Windows or Linux.
        """
        mafft_bin = self._ensure_mafft_installed()
        flag = "--adjustdirection" if adjustdirection else ""
        cmd = f"\"{mafft_bin}\" {flag} \"{src}\" > \"{des}\""
        logger.info(f"Running MAFFT: {cmd}")
        self._exec(cmd, suppress_output=True)

    def mafft_consensus(
        self,
        src: str,
        des: str,
        minimal_reads: int = 0,
        input_format: str = "fas",
        max_reads: int = 10000,
        adjustdirection: bool = False
    ) -> str:
        """
        For each file in 'src', align up to max_reads sequences (randomly sampled if needed),
        build a naive consensus, and save both alignment and consensus in 'des'.
        Returns the absolute path to 'des'.
        """
        os.makedirs(des, exist_ok=True)
        abs_des = os.path.abspath(des)

        for entry in os.scandir(src):
            SampleID, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            io_fmt = self._check_input_output(input_format, "fasta")
            if not entry.is_file():
                continue

            logger.info(f"Working on '{entry.name}'")
            if ext not in io_fmt["input"]:
                logger.info(f"Skipping '{entry.name}': not in {io_fmt['input']} format.")
                continue

            con_path = os.path.join(abs_des, f"con_{SampleID}.fas")
            if os.path.isfile(con_path):
                logger.info(f"'{entry.name}' already processed, skipping.")
                continue

            # Determine sequence source and count
            if ext in self.fasta_ext:
                seq_num = sum(1 for _ in self._fasta_reader(open(entry.path, "r")))
                fas_path = entry.path
            elif ext in self.fastq_ext:
                seq_num = sum(1 for _ in self._fastq_reader(open(entry.path, "r")))
                self._clean_temp()
                fas_path = os.path.join(self.TEMP, "from_fastq.fas")
                self._fastq_to_fasta(entry.path, fas_path)
            else:
                continue

            if seq_num < minimal_reads:
                logger.info(
                    f"Skipping '{entry.name}': only {seq_num} reads (< {minimal_reads})."
                )
                continue

            # Random sampling if too many reads
            if 0 < max_reads < seq_num:
                logger.info(
                    f"'{entry.name}' has {seq_num} reads (> {max_reads}), sampling {max_reads} reads."
                )
                records = list(self._fasta_reader(open(fas_path, "r")))
                sampled = random_sample(records, max_reads)
                with open(fas_path, "w") as fh:
                    for rec in sampled:
                        fh.write(f">{rec['title']}\n{rec['seq']}\n")

            # Align and build consensus
            aln_path = os.path.join(abs_des, f"aln_{SampleID}.fas")
            self._mafft(fas_path, aln_path, adjustdirection=adjustdirection)
            self._naive_consensus(aln_path, con_path, SampleID)

        return abs_des

    def _gblock(self, fas: str) -> str:
        """
        Run Gblocks on the input FASTA file 'fas' and return the path to the processed file.
        Gblocks parameters: -t=d (DNA), -b5=a (keep all gaps), -e=gblo (output extension).
        """
        # Copy input file to temp folder
        try:
            os.makedirs(self.TEMP, exist_ok=True)
            dst = os.path.join(self.TEMP, os.path.basename(fas))
            shutil.copy(fas, dst)
            fas = dst
            logger.info(f"Copied '{fas}' to temp, running Gblocks.")
        except Exception as e:
            logger.warning(f"Failed to copy '{fas}' to temp: {e}")

        gblock_bin = os.path.join(self._lib_path(), "bin", "Gblocks")
        cmd = [gblock_bin, fas, "-t=d", "-b5=a", "-e=gblo"]
        cmd_line = " ".join(cmd)
        logger.info(f"Running Gblocks: {cmd_line}")
        self._exec(cmd_line, suppress_output=True)

        output_path = f"{fas}gblo"
        # Remove spaces in sequences
        processed_seqs: List[str] = []
        try:
            with open(output_path, "r") as handle:
                for rec in self._fasta_reader(handle):
                    seq = rec["seq"].replace(" ", "")
                    processed_seqs.append(f">{rec['title']}\n{seq}\n")
            with open(output_path, "w") as handle:
                handle.writelines(processed_seqs)
        except Exception as e:
            logger.error(f"Error processing Gblocks output '{output_path}': {e}")
            raise

        return output_path

    def diplod_site(self, MSA: np.ndarray, threshold: float = 0.2) -> Tuple[np.ndarray, List[int]]:
        """
        Identify diploid sites in a multiple sequence alignment (MSA).
        Returns a sub-MSA containing only diploid columns and the list of diploid column indices.
        """
        diploid_positions: List[int] = []
        freqs: List[List[float]] = []

        for i in range(MSA.shape[1]):
            column = MSA[:, i]
            c = Counter(column)
            ratios: List[float] = [(c[n] / MSA.shape[0]) for n in "ATCG"]
            freqs.append(ratios)
            # If the second highest frequency exceeds threshold, mark as diploid site
            if sorted(ratios)[-2] > threshold:
                diploid_positions.append(i)

        freqs_array = np.array(freqs)
        MSA_diploid = MSA[:, diploid_positions]
        return MSA_diploid, diploid_positions

    def diploid_cluster(
        self,
        src: str,
        des: str,
        input_format: str = "fas",
        output_format: str = "fas"
    ) -> None:
        """
        Perform fine-grained clustering on diploid sites after initial clustering.
        Inputs:
            src: directory of aligned FASTA files (must start with 'aln_').
            des: output directory to store new clusters and consensus sequences.
            input_format: file extension of input FASTA (default 'fas').
            output_format: desired output FASTA extension (default 'fas').
        """
        io_format = self._check_input_output(input_format, output_format)
        os.makedirs(des, exist_ok=True)

        for entry in os.scandir(src):
            if not entry.is_file():
                continue
            filename, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            if ext not in io_format["input"]:
                logger.info(f"Skipping '{entry.name}': not in {io_format['input']} format.")
                continue
            if not entry.name.startswith("aln_"):
                logger.info(f"Skipping '{entry.name}': does not start with 'aln_'.")
                continue

            logger.info(f"Processing '{entry.name}'")
            FAS = entry.path
            parts = entry.name.split("_")
            try:
                SampleID = parts[1]
                original_cluster = parts[3]
                original_prefix = "_".join(parts[0:3])
            except IndexError:
                logger.warning(
                    f"Failed to parse '{entry.name}'. "
                    "Filename must be 'aln_[SampleID]_cluster_[number]_r[number].{ext}'"
                )
                continue

            # Read MSA_gblocked into a numpy array
            records = list(self._fasta_reader(open(FAS, "r")))
            titles = [rec["title"] for rec in records]
            seqs = [list(rec["seq"].upper()) for rec in records]
            try:
                MSA_gblocked = np.array(seqs)
            except ValueError:
                logger.error("Failed to read MSA: sequences have differing lengths.")
                continue

            # Identify diploid sites and extract sub-MSA
            MSA_diploid, diploid_positions = self.diplod_site(MSA_gblocked, 0.2)

            # Prepare temp directories
            temp_diploid = os.path.join(self.TEMP, "MSA_diploid")
            temp_labeled = os.path.join(self.TEMP, "MSA_labeled")
            temp_cluster = os.path.join(self.TEMP, "MSA_diploid_cluster")
            for d in [temp_diploid, temp_labeled, temp_cluster]:
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d, exist_ok=True)

            # Save MSA_diploid
            diploid_fas = os.path.join(temp_diploid, "MSA_diploid.fas")
            with open(diploid_fas, "w") as f_out:
                for idx, row in enumerate(MSA_diploid):
                    f_out.write(f">{titles[idx]}\n{''.join(row)}\n")

            # Label diploid sites (lowercase) in MSA_gblocked and save
            labeled = MSA_gblocked.copy()
            for pos in diploid_positions:
                labeled[:, pos] = np.char.lower(labeled[:, pos])
            labeled_fas = os.path.join(temp_labeled, "MSA_labeled.fas")
            with open(labeled_fas, "w") as f_out:
                for idx, row in enumerate(labeled):
                    f_out.write(f">{titles[idx]}\n{''.join(row)}\n")

            # Cluster on diploid sites using HDBSCAN (via self.hdbscan)
            self.hdbscan(
                temp_diploid,
                temp_cluster,
                input_format="fas",
                output_format="fas",
                min_cluster_size=0.1,
                mds=False
            )

            # Read original sequences to reconstruct clusters
            original_sequences: Dict[str, str] = {
                rec["title"]: rec["seq"]
                for rec in self._fasta_reader(open(FAS, "r"))
            }

            for cluster_file in os.scandir(temp_cluster):
                if not cluster_file.name.endswith(f".{ext}"):
                    continue
                cluster_number = cluster_file.name.split("_")[3]
                reads_count = 0
                output_records: List[str] = []
                for rec in self._fasta_reader(open(cluster_file.path, "r")):
                    title = rec["title"]
                    seq = original_sequences.get(title, "")
                    output_records.append(f">{title}\n{seq}\n")
                    reads_count += 1

                clusterID = f"{SampleID}_cluster_{original_cluster}-{cluster_number}_r{reads_count}"
                new_aln = f"aln_{clusterID}.{io_format['output']['fasta']}"
                new_con = f"con_{clusterID}.{io_format['output']['fasta']}"

                aln_path = os.path.join(des, new_aln)
                with open(aln_path, "w") as fw:
                    fw.writelines(output_records)

                con_path = os.path.join(des, new_con)
                self._naive_consensus(aln_path, con_path, clusterID)

    def orientation(
        self,
        src: str,
        des: str,
        input_format: str = "fastq",
        output_format: str = "both",
        BARCODE_INDEX_FILE: str = "",
        FwPrimer: str = "FwPrimer",
        RvPrimer: str = "RvPrimer",
        search_range: int = 500
    ) -> str:
        """
        Ensure all reads have the correct orientation based on forward/reverse primers.
        Inputs:
            src: directory with input files named {SampleID}.fastq or .fas
            des: output directory for re-oriented files
            BARCODE_INDEX_FILE: TSV/CSV containing columns SampleID, FwPrimer, RvPrimer
            search_range: number of bases at start of read to search for primer
        Returns:
            Path to output directory.
        """
        io_format = self._check_input_output(input_format, output_format)
        os.makedirs(des, exist_ok=True)

        # Load barcode index
        if BARCODE_INDEX_FILE.endswith(".tsv"):
            bar_idx = pd.read_csv(BARCODE_INDEX_FILE, sep="\t")
        elif BARCODE_INDEX_FILE.endswith(".csv"):
            bar_idx = pd.read_csv(BARCODE_INDEX_FILE)
        elif BARCODE_INDEX_FILE.endswith(".xlsx"):
            bar_idx = pd.read_excel(BARCODE_INDEX_FILE)
        else:
            logger.error(f"Unsupported barcode index file format: {BARCODE_INDEX_FILE}")
            return ""

        for entry in os.scandir(src):
            if not entry.is_file():
                continue
            filename, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            if ext not in io_format["input"]:
                logger.info(f"Skipping '{entry.name}': not in {io_format['input']} format.")
                continue

            logger.info(f"Processing '{entry.name}'")
            try:
                sample_row = bar_idx[bar_idx["SampleID"].astype(str) == filename].iloc[0]
                F = sample_row[FwPrimer]
                R = sample_row[RvPrimer]
            except (IndexError, KeyError):
                logger.warning(f"Sample '{filename}' not found in barcode file, skipping.")
                continue

            output_fasta: Optional[TextIO] = None
            output_fastq: Optional[TextIO] = None
            if "fasta" in io_format["output"]:
                path_fa = os.path.join(des, f"{filename}.{io_format['output']['fasta']}")
                output_fasta = open(path_fa, "w")
            if "fastq" in io_format["output"]:
                path_fq = os.path.join(des, f"{filename}.{io_format['output']['fastq']}")
                output_fastq = open(path_fq, "w")

            with open(entry.path, "r") as handle:
                if entry.name.endswith(".fastq"):
                    records = self._fastq_reader(handle)
                else:
                    records = self._fasta_reader(handle)

                for record in records:
                    seq_upper = record["seq"].upper()[:search_range]
                    aln_f = edlib.align(F.upper(), seq_upper, mode="HW", task="locations")
                    aln_r = edlib.align(R.upper(), seq_upper, mode="HW", task="locations")
                    if aln_f["editDistance"] > aln_r["editDistance"]:
                        record["seq"] = self._reverse_complement(record["seq"])

                    if output_fasta:
                        output_fasta.write(f">{record['title']}\n{record['seq']}\n")
                    if output_fastq:
                        qual = record.get("qual", "")
                        output_fastq.write(f"@{record['title']}\n{record['seq']}\n+\n{qual}\n")

            if output_fasta:
                output_fasta.close()
            if output_fastq:
                output_fastq.close()

        return des
 
    def _naive_consensus(self, src: str, des: str, title: str) -> None:
        """
        Build a naive consensus from an aligned FASTA file 'src' and write to 'des'.
        Consensus takes the most common non-gap base at each column.
        """
        try:
            records = list(self._fasta_reader(open(src, "r")))
            length = len(records[0]["seq"])
            consensus = []
            for i in range(length):
                column = [r["seq"][i] for r in records]
                most_common = Counter(column).most_common(1)[0][0]
                if most_common != "-":
                    consensus.append(most_common)
            with open(des, "w") as out:
                out.write(f">{title}\n{''.join(consensus)}")
            logger.info(f"Wrote consensus to '{des}'.")
        except Exception as e:
            logger.error(f"Error building consensus from '{src}': {e}")

    def _get_sample_id_single(
        self,
        seq: str,
        barcode_hash_table: Dict[str, Dict[str, Union[str, float]]],
        search_range: int = 150,
        mismatch_ratio_f: float = 0.15,
        mismatch_ratio_r: float = 0.15,
    ) -> Tuple[List[str], List[bool], List[str]]:
        """
        Identify matching SampleIDs for one sequence read based on its barcode.

        Returns:
            ids: List of SampleIDs that matched at least the forward barcode.
            integrity: Boolean list indicating if reverse anchor also matched for each ID.
            seqs: List of sequences with matched barcode regions lowercased.
        """
        ids: List[str] = []
        integrity: List[bool] = []
        seqs: List[str] = []

        seq = seq.upper()
        seqF = seq[:search_range]
        seqR = seq[-search_range:]

        for sample_id, info in barcode_hash_table.items():
            fw_index = str(info["FwIndex"]).upper()
            rv_anchor = str(info["RvAnchor"]).upper()

            fw_align = edlib.align(
                fw_index,
                seqF,
                mode="HW",
                k=int(len(fw_index) * mismatch_ratio_f),
                task="locations",
            )
            rv_align = edlib.align(
                rv_anchor,
                seqR,
                mode="HW",
                k=int(len(rv_anchor) * mismatch_ratio_r),
                task="locations",
            )

            if fw_align["editDistance"] != -1:
                # Mark forward match region lowercase
                f_start, f_end = fw_align["locations"][0]
                seqF_marked = (
                    seqF[:f_start]
                    + seqF[f_start : f_end + 1].lower()
                    + seqF[f_end + 1 :]
                )

                seqR_marked = seqR
                if rv_align["editDistance"] != -1 and rv_anchor:
                    # Mark reverse match region lowercase
                    r_start, r_end = rv_align["locations"][0]
                    seqR_marked = (
                        seqR[:r_start]
                        + seqR[r_start : r_end + 1].lower()
                        + seqR[r_end + 1 :]
                    )
                    integrity.append(True)
                else:
                    integrity.append(False)

                ids.append(sample_id)
                seqs.append(seqF_marked + seq[search_range : -search_range] + seqR_marked)

        return ids, integrity, seqs
    def singlebar(
        self,
        src: str,
        des: str,
        BARCODE_INDEX_FILE: str,
        mismatch_ratio_f: float = 0.15,
        mismatch_ratio_r: float = 0.15,
        expected_length_variation: float = 0.3,
        search_range: int = 150,
        rvc_rvanchor: bool = False,
        input_format: str = "fastq",
        output_format: str = "both",
    ) -> str:
        """
        Demultiplex a single FASTQ/FASTA file by barcodes.

        Args:
            src: Path to the input FASTQ/FASTA file.
            des: Directory where demultiplexed files and stats will be saved.
            BARCODE_INDEX_FILE: TSV or CSV containing columns
                SampleID, FwIndex, RvAnchor, ExpectedLength.
            mismatch_ratio_f: Allowed forward barcode mismatch ratio.
            mismatch_ratio_r: Allowed reverse anchor mismatch ratio.
            expected_length_variation: Fractional variation allowed in total read length.
            search_range: Number of bases from each end to search for barcodes.
            rvc_rvanchor: If True, use reverse complement of RvAnchor.
            input_format: "fastq" or "fasta".
            output_format: "fastq", "fasta", or "both".

        Returns:
            des: Output directory path.
        """
        io_format = self._check_input_output(
            input_format=input_format, output_format=output_format
        )

        # Determine delimiter for barcode index file
        # Load barcode index
        if BARCODE_INDEX_FILE.endswith(".tsv"):
            barcode_df = pd.read_csv(BARCODE_INDEX_FILE, sep="\t")
        elif BARCODE_INDEX_FILE.endswith(".csv"):
            barcode_df = pd.read_csv(BARCODE_INDEX_FILE)
        elif BARCODE_INDEX_FILE.endswith(".xlsx"):
            barcode_df = pd.read_excel(BARCODE_INDEX_FILE)
        else:
            logger.error(f"Unsupported barcode index file format: {BARCODE_INDEX_FILE}")
            return ""


        required_cols = {"SampleID", "FwIndex", "RvAnchor", "ExpectedLength"}
        if not required_cols.issubset(barcode_df.columns):
            raise ValueError(
                f"BARCODE_INDEX_FILE must have columns: {', '.join(required_cols)}"
            )
        logger.info("Loaded BARCODE_INDEX_FILE")

        # Build hash table: SampleID -> {FwIndex, RvAnchor, ExpectedLength}
        barcode_hash_table: Dict[str, Dict[str, Union[str, float]]] = {}
        for _, row in barcode_df.iterrows():
            rv_anchor_val = "" if pd.isnull(row["RvAnchor"]) else str(row["RvAnchor"])
            if rvc_rvanchor and rv_anchor_val:
                rv_anchor_val = self._reverse_complement(rv_anchor_val)
            barcode_hash_table[str(row["SampleID"])] = {
                "FwIndex": str(row["FwIndex"]),
                "RvAnchor": rv_anchor_val,
                "ExpectedLength": float(row["ExpectedLength"]),
            }

        # Prepare containers for reads
        pool: Dict[str, List[Dict[str, str]]] = {
            "UNKNOWN": [],
            "MULTIPLE": [],
            "TRUNCATED": [],
            "IncorrectLength": [],
        }
        for sample_id in barcode_hash_table:
            pool[sample_id] = []

        # Determine file extension
        _, ext = os.path.splitext(src)
        ext = ext.lstrip(".").lower()

        with open(src, "r") as infile:
            if ext in self.fastq_ext and input_format in self.fastq_ext:
                reader = self._fastq_reader(infile)
            elif ext in self.fasta_ext and input_format in self.fasta_ext:
                reader = self._fasta_reader(infile)
            else:
                raise ValueError(
                    f"Input file extension '{ext}' does not match input format '{input_format}'."
                )

            total = 0
            for record in reader:
                ids, integrity_flags, seqs = self._get_sample_id_single(
                    record["seq"],
                    barcode_hash_table,
                    search_range,
                    mismatch_ratio_f,
                    mismatch_ratio_r,
                )

                if len(ids) == 1:
                    sample_id = ids[0]
                    rv_anchor_seq = barcode_hash_table[sample_id]["RvAnchor"]
                    # If RvAnchor is empty, skip anchor check
                    if rv_anchor_seq == "":
                        integrity_flag = True
                        seq_to_use = record["seq"]
                    else:
                        integrity_flag = integrity_flags[0]
                        seq_to_use = seqs[0]

                    record["seq"] = seq_to_use
                    if not integrity_flag:
                        pool["TRUNCATED"].append(record)
                    else:
                        expected_len = barcode_hash_table[sample_id]["ExpectedLength"]
                        length_diff = abs(len(seq_to_use) - expected_len)
                        if length_diff <= expected_len * expected_length_variation:
                            pool[sample_id].append(record)
                        else:
                            pool["IncorrectLength"].append(record)

                elif len(ids) > 1:
                    pool["MULTIPLE"].append(record)
                else:
                    pool["UNKNOWN"].append(record)

                total += 1
                if total % 10000 == 0:
                    logger.info(f"Parsed {total} reads")

        os.makedirs(des, exist_ok=True)
        os.makedirs(os.path.join(des, "trash"), exist_ok=True)

        # Build stats DataFrame
        stat_rows: List[Dict[str, Union[str, int]]] = []
        for key, records in pool.items():
            stat_rows.append({"SampleID": key, "Count": len(records)})
        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(os.path.join(des, "2_Singlebar_stat.csv"), index=False)

        # Write out demultiplexed files
        for key, records in pool.items():
            if key in {"UNKNOWN", "MULTIPLE", "TRUNCATED", "IncorrectLength"}:
                subdir = os.path.join(des, "trash", key)
            else:
                subdir = os.path.join(des, key)

            os.makedirs(os.path.dirname(subdir), exist_ok=True)

            if "fastq" in io_format["output"]:
                fastq_path = f"{subdir}.{io_format['output']['fastq']}"
                with open(fastq_path, "w") as fq_out:
                    for rec in records:
                        fq_out.write(f"@{rec['title']}\n")
                        fq_out.write(f"{rec['seq']}\n")
                        fq_out.write("+\n")
                        fq_out.write(f"{rec.get('qual', '')}\n")

            if "fasta" in io_format["output"]:
                fasta_path = f"{subdir}.{io_format['output']['fasta']}"
                with open(fasta_path, "w") as fa_out:
                    for rec in records:
                        fa_out.write(f">{rec['title']}\n")
                        fa_out.write(f"{rec['seq']}\n")

        failed = (
            len(pool["UNKNOWN"])
            + len(pool["MULTIPLE"])
            + len(pool["TRUNCATED"])
            + len(pool["IncorrectLength"])
        )
        success_pct = (total - failed) / total * 100 if total > 0 else 0.0
        logger.info(f"{total - failed}/{total} ({success_pct:.2f}%) reads were demultiplexed successfully")

        return des

    
    def double_demultiplex(
        self,
        src: str,
        des: str,
        BARCODE_INDEX_FILE: str,
        primers: List[str] = ["FwPrimer", "RvPrimer"],
        mismatch_ratio_f: float = 0.15,
        mismatch_ratio_r: float = 0.15,
        expected_length_variation: float = 0.3,
        search_range: int = 150,
        rvc_rvanchor: bool = False,
        input_format: str = "fastq",
        output_format: str = "both",
    ) -> str:
        """
        Single FASTQ demultiplexing into separate sample files (and optional FASTA output).

        Args:
            src: Path to the input FASTQ/FASTA file (e.g., "all.fastq").
            des: Output directory where per-sample files will be written.
            BARCODE_INDEX_FILE: CSV or TSV with columns:
                SampleID, FwIndex, FwPrimer, RvAnchor, RvPrimer, ExpectedLength.
            primers: List of column names for primers (default ["FwPrimer", "RvPrimer"]).
            mismatch_ratio_f: Allowed error ratio for FwIndex (fraction of barcode length).
            mismatch_ratio_r: Allowed error ratio for RvPrimer (fraction of barcode length).
            expected_length_variation: Allowed deviation fraction around ExpectedLength.
            search_range: How many bases from each end to search for barcodes (default 150).
            rvc_rvanchor: If True, use reverse-complement of RvAnchor from index file.
            input_format: "fastq" or "fasta".
            output_format: "fastq", "fasta", or "both".

        Returns:
            The output directory (`des`) where demultiplexed files are placed.
        """
        # Validate formats
        io_format = self._check_input_output(input_format=input_format, output_format=output_format)

        # Load barcode index
        if BARCODE_INDEX_FILE.endswith(".tsv"):
            barcode_df = pd.read_csv(BARCODE_INDEX_FILE, sep="\t")
        elif BARCODE_INDEX_FILE.endswith(".csv"):
            barcode_df = pd.read_csv(BARCODE_INDEX_FILE)
        elif BARCODE_INDEX_FILE.endswith(".xlsx"):
            barcode_df = pd.read_excel(BARCODE_INDEX_FILE)
        else:
            logger.error(f"Unsupported barcode index file format: {BARCODE_INDEX_FILE}")
            return ""



        # Required columns: SampleID, FwIndex, FwPrimer, RvAnchor, RvPrimer, ExpectedLength
        required_cols = ["SampleID", "FwIndex", "FwPrimer", "RvAnchor", "RvPrimer", "ExpectedLength"]
        required_cols.extend(primers)
        missing = [c for c in required_cols if c not in barcode_df.columns]
        if missing:
            raise ValueError(f"BARCODE_INDEX_FILE is missing columns: {', '.join(missing)}")

        # Build hash tables per barcode type (FwIndex and primers)
        barcode_hash_tables: Dict[str, Dict[str, Dict[str, Union[str, float]]]] = {}
        for col in ["FwIndex"] + primers:
            table: Dict[str, Dict[str, Union[str, float]]] = {}
            for _, row in barcode_df.iterrows():
                sample_id = row["SampleID"]
                # Optionally reverse-complement RvAnchor before storing
                rv_anchor = str(row["RvAnchor"])
                if rvc_rvanchor:
                    rv_anchor = self._reverse_complement(rv_anchor)
                table[sample_id] = {
                    "FwIndex": str(row[col]).upper(),
                    "RvAnchor": rv_anchor,
                    "ExpectedLength": float(row["ExpectedLength"]),
                }
            barcode_hash_tables[col] = table

        # Prepare pools for each sample, plus special bins
        pool: Dict[str, List[Dict[str, str]]] = {
            sample_id: [] for sample_id in barcode_hash_tables["FwIndex"]
        }
        for special in ["UNKNOWN", "MULTIPLE", "TRUNCATED", "IncorrectLength"]:
            pool[special] = []

        # Determine file extension of src
        _, ext = os.path.splitext(src)
        ext = ext.lstrip(".")

        # Read input file (FASTQ or FASTA)
        with open(src, "r") as handle:
            if ext in self.fastq_ext and input_format in self.fastq_ext:
                reader = self._fastq_reader(handle)
            elif ext in self.fasta_ext and input_format in self.fasta_ext:
                reader = self._fasta_reader(handle)
            else:
                raise ValueError(f"Input file extension '{ext}' does not match declared format '{input_format}'")

            counter = 0
            for record in reader:
                counter += 1
                if counter % 1000 == 0:
                    logger.info(f"Parsed {counter} records", extra={"end": "\r"})

                # First-pass: demultiplex by FwIndex
                fw_table = barcode_hash_tables["FwIndex"]
                ids_from_fw, integrity_list, seqs_from_fw = self._get_sample_id_single(
                    record["seq"],
                    fw_table,
                    search_range,
                    mismatch_ratio_f,
                    mismatch_ratio_r,
                )

                if not ids_from_fw:
                    pool["UNKNOWN"].append(record)
                    continue

                # Second-pass: demultiplex by primers
                primer_ids: List[str] = []
                for primer_col in primers:
                    primer_table = barcode_hash_tables[primer_col]
                    ids, _, _ = self._get_sample_id_single(
                        record["seq"],
                        primer_table,
                        search_range,
                        mismatch_ratio_f,
                        mismatch_ratio_r,
                    )
                    primer_ids.extend(ids)

                # Intersection of FwIndex IDs and primer IDs
                final_ids = list(set(ids_from_fw).intersection(primer_ids))

                if len(final_ids) == 1:
                    sample_id = final_ids[0]
                    seq_candidate = seqs_from_fw[0]
                    if not integrity_list[0]:
                        pool["TRUNCATED"].append(record)
                    else:
                        exp_len = barcode_hash_tables["FwIndex"][sample_id]["ExpectedLength"]
                        observed_len = len(seq_candidate)
                        if (observed_len - exp_len) ** 2 < (exp_len * expected_length_variation) ** 2:
                            record["seq"] = seq_candidate
                            pool[sample_id].append(record)
                        else:
                            pool["IncorrectLength"].append(record)
                elif len(final_ids) > 1:
                    pool["MULTIPLE"].append(record)
                else:
                    pool["UNKNOWN"].append(record)

        # Create output directories
        os.makedirs(des, exist_ok=True)
        trash_dir = os.path.join(des, "trash")
        os.makedirs(trash_dir, exist_ok=True)

        # Collect stats
        stat_rows: List[Dict[str, Union[str, int]]] = []
        for bin_name, records in pool.items():
            stat_rows.append({"SampleID": bin_name, "Count": len(records)})
            if bin_name in ["MULTIPLE", "UNKNOWN", "TRUNCATED", "IncorrectLength"]:
                out_prefix = os.path.join(trash_dir, bin_name)
            else:
                out_prefix = os.path.join(des, bin_name)

            # Write FASTQ
            if "fastq" in io_format["output"]:
                fastq_path = f"{out_prefix}.{io_format['output']['fastq']}"
                with open(fastq_path, "w") as fq_handle:
                    for rec in records:
                        fq_handle.write(f"@{rec['title']}\n{rec['seq']}\n+\n{rec['qual']}\n")

            # Write FASTA
            if "fasta" in io_format["output"]:
                fasta_path = f"{out_prefix}.{io_format['output']['fasta']}"
                with open(fasta_path, "w") as fa_handle:
                    for rec in records:
                        fa_handle.write(f">{rec['title']}\n{rec['seq']}\n")

        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(os.path.join(des, "2_Doublebar_stat.csv"), index=False)

        total_reads = counter
        failed_reads = (
            len(pool["UNKNOWN"])
            + len(pool["MULTIPLE"])
            + len(pool["TRUNCATED"])
            + len(pool["IncorrectLength"])
        )
        success_pct = (total_reads - failed_reads) / total_reads * 100 if total_reads else 0.0
        logger.info(
            f"{total_reads - failed_reads}/{total_reads} "
            f"({success_pct:.2f}%) reads were demultiplexed successfully."
        )
        return des

    def _get_sample_id_dual(
        self,
        seq: str,
        barcode_hash_table: Dict[str, Dict[str, Union[str, float]]],
        mismatch_ratio_f: float = 0.15,
        mismatch_ratio_r: float = 0.15,
    ) -> Tuple[List[str], List[bool], List[str]]:
        """
        Identify sample IDs based on both forward and reverse index matching.
        Returns (matched_ids, integrity_flags, trimmed_seqs).

        Args:
            seq: Input read sequence (string).
            barcode_hash_table: Dict mapping SampleID -> {"FwIndex": str, "RvIndex": str, "ExpectedLength": float}.
            mismatch_ratio_f: Fractional edit-distance threshold for forward index.
            mismatch_ratio_r: Fractional edit-distance threshold for reverse index.

        Returns:
            matched_ids: List of sample IDs for which both forward and reverse indices match.
            integrity_flags: List of booleans indicating if both indices were found intact.
            trimmed_seqs: List of sequences trimmed around the found barcode regions.
        """
        seq_upper = seq.upper()
        seqF = seq_upper[:150]
        seqR = seq_upper[-150:]
        seq_REV = self._reverse_complement(seq_upper)
        seq_REV_F = seq_REV[:150]
        seq_REV_R = seq_REV[-150:]

        matched_ids: List[str] = []
        integrity_flags: List[bool] = []
        trimmed_seqs: List[str] = []

        for sample_id, info in barcode_hash_table.items():
            fw_index = str(info["FwIndex"]).upper()
            rv_index = str(info["RvIndex"]).upper()
            rv_index_rc = self._reverse_complement(rv_index)

            # Search forward index in front segment
            fw_hit = edlib.align(
                fw_index,
                seqF,
                mode="HW",
                k=int(len(fw_index) * mismatch_ratio_f),
                task="locations",
            )
            # Search reverse index in rear segment
            rv_hit = edlib.align(
                rv_index,
                seqR,
                mode="HW",
                k=int(len(rv_index) * mismatch_ratio_r),
                task="locations",
            )
            # Also check reverse complement scenario
            fw_hit_rc = edlib.align(
                fw_index,
                seq_REV_F,
                mode="HW",
                k=int(len(fw_index) * mismatch_ratio_f),
                task="locations",
            )
            rv_hit_rc = edlib.align(
                rv_index_rc,
                seq_REV_R,
                mode="HW",
                k=int(len(rv_index_rc) * mismatch_ratio_r),
                task="locations",
            )

            # If forward and reverse indices match in original orientation
            if fw_hit["editDistance"] != -1 and rv_hit["editDistance"] != -1:
                f_start, f_end = fw_hit["locations"][0]
                r_start, r_end = rv_hit["locations"][0]
                # Mark located regions as lowercase
                seqF = seqF[:f_start] + seqF[f_start:f_end].lower() + seqF[f_end:]
                seqR = seqR[:r_start] + seqR[r_start:r_end].lower() + seqR[r_end:]
                trimmed = seqF + seq_upper[150:-150] + seqR
                matched_ids.append(sample_id)
                integrity_flags.append(True)
                trimmed_seqs.append(trimmed)

            # If both match on reverse-complement strand
            elif fw_hit_rc["editDistance"] != -1 and rv_hit_rc["editDistance"] != -1:
                f_start_rc, f_end_rc = fw_hit_rc["locations"][0]
                r_start_rc, r_end_rc = rv_hit_rc["locations"][0]
                seq_REV_F = (
                    seq_REV_F[:f_start_rc]
                    + seq_REV_F[f_start_rc:f_end_rc].lower()
                    + seq_REV_F[f_end_rc:]
                )
                seq_REV_R = (
                    seq_REV_R[:r_start_rc]
                    + seq_REV_R[r_start_rc:r_end_rc].lower()
                    + seq_REV_R[r_end_rc:]
                )
                trimmed_rc = seq_REV_F + seq_upper[150:-150] + seq_REV_R
                matched_ids.append(sample_id)
                integrity_flags.append(True)
                trimmed_seqs.append(trimmed_rc)

        return matched_ids, integrity_flags, trimmed_seqs

    def dualbar(
        self,
        src: str,
        des: str,
        BARCODE_INDEX_FILE: str,
        mismatch_ratio_f: float = 0.15,
        mismatch_ratio_r: float = 0.15,
        expected_length_variation: float = 0.3,
    ) -> str:
        """
        Dual-index demultiplexing on a single FASTQ file.

        Args:
            src: Path to input FASTQ (e.g., "./data/1_Nanofilt/all.fastq").
            des: Output directory for demultiplexed files.
            BARCODE_INDEX_FILE: TSV or CSV with columns:
                SampleID, FwIndex, RvIndex, ExpectedLength.
            mismatch_ratio_f: Allowed error ratio for forward index.
            mismatch_ratio_r: Allowed error ratio for reverse index.
            expected_length_variation: Allowed length deviation for trimmed read.

        Returns:
            The output directory (`des`) after writing per-sample files.
        """
        # Load index table
        if BARCODE_INDEX_FILE.endswith(".tsv"):
            sep = "\t"
        elif BARCODE_INDEX_FILE.endswith(".csv"):
            sep = ","
        else:
            raise ValueError("Barcode index file must be .tsv or .csv")

        barcode_df = pd.read_csv(BARCODE_INDEX_FILE, sep=sep)
        required = ["SampleID", "FwIndex", "RvIndex", "ExpectedLength"]
        missing_cols = [col for col in required if col not in barcode_df.columns]
        if missing_cols:
            raise ValueError(f"Barcode index file is missing: {', '.join(missing_cols)}")

        logger.info("Loaded barcode index file.")

        # Create hash table
        barcode_hash_table: Dict[str, Dict[str, Union[str, float]]] = {}
        for _, row in barcode_df.iterrows():
            sample_id = row["SampleID"]
            barcode_hash_table[sample_id] = {
                "FwIndex": str(row["FwIndex"]).upper(),
                "RvIndex": str(row["RvIndex"]).upper(),
                "ExpectedLength": float(row["ExpectedLength"]),
            }

        # Initialize pools
        pool: Dict[str, List[Dict[str, str]]] = {
            sid: [] for sid in barcode_hash_table
        }
        for special in ["Unknown", "Multiple", "IncorrectLength"]:
            pool[special] = []

        counter = 0
        with open(src, "r") as handle:
            for record in self._fastq_reader(handle):
                seq = record["seq"]
                ids, trimmed_seqs = self._get_sample_id_dual(
                    seq, barcode_hash_table, mismatch_ratio_f, mismatch_ratio_r
                )
                if len(ids) == 1:
                    sample_id = ids[0]
                    trimmed = trimmed_seqs[0]
                    exp_len = barcode_hash_table[sample_id]["ExpectedLength"]
                    if (len(trimmed) - exp_len) ** 2 < (exp_len * expected_length_variation) ** 2:
                        record["seq"] = trimmed
                        pool[sample_id].append(record)
                    else:
                        pool["IncorrectLength"].append(record)
                elif len(ids) > 1:
                    pool["Multiple"].append(record)
                else:
                    pool["Unknown"].append(record)

                counter += 1
                if counter % 10000 == 0:
                    logger.info(f"Processed {counter} reads")

        os.makedirs(des, exist_ok=True)
        trash_dir = os.path.join(des, "trash")
        os.makedirs(trash_dir, exist_ok=True)

        stat_rows: List[Dict[str, Union[str, int]]] = []
        for bin_name, records in pool.items():
            stat_rows.append({"SampleID": bin_name, "Count": len(records)})
            if bin_name in ["Multiple", "Unknown", "IncorrectLength"]:
                out_prefix = os.path.join(trash_dir, bin_name)
            else:
                out_prefix = os.path.join(des, bin_name)

            fastq_path = f"{out_prefix}.fastq"
            with open(fastq_path, "w") as fq_handle:
                for rec in records:
                    fq_handle.write(f"@{rec['title']}\n{rec['seq']}\n+\n{rec['qual']}\n")

            fasta_path = f"{out_prefix}.fas"
            with open(fasta_path, "w") as fa_handle:
                for rec in records:
                    fa_handle.write(f">{rec['title']}\n{rec['seq']}\n")

        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(os.path.join(des, "2_Dualbar_stat.csv"), index=False)

        failed = len(pool["Unknown"]) + len(pool["Multiple"]) + len(pool["IncorrectLength"])
        success_pct = (counter - failed) / counter * 100 if counter else 0.0
        logger.info(f"{counter - failed}/{counter} ({success_pct:.2f}%) reads demultiplexed successfully.")
        return des

    def combine_fastq(self, src: str, des: str, name: str = "all.fastq") -> str:
        """
        Concatenate all .fastq.gz files in `src` directory into a single FASTQ file.

        Args:
            src: Directory containing .fastq.gz files.
            des: Directory to write the combined FASTQ.
            name: Output filename (default "all.fastq").

        Returns:
            Path to the combined FASTQ file.
        """
        os.makedirs(des, exist_ok=True)
        out_path = os.path.join(des, name)

        with open(out_path, "w") as outfile:
            for entry in os.scandir(src):
                if entry.name.endswith(".fastq.gz"):
                    logger.info(f"Found FASTQ file: {entry.name}")
                    with gzip.open(entry.path, "rt") as infile:
                        for line in infile:
                            outfile.write(line)
        return out_path

    def nanofilt(
        self,
        src: str,
        des: str,
        name: str = "all.fastq",
        NANOFILT_QSCORE: int = 8,
        NANOFILT_MIN_LEN: int = 400,
        NANOFILT_MAX_LEN: int = 8000
    ) -> str:
        """
        Filter reads using NanoFilt with specified quality and length thresholds.
        Returns the path to the filtered FASTQ file.
        """
        os.makedirs(des, exist_ok=True)
        logger.info("Starting NanoFilt...")
        output_path = os.path.join(des, name)
        cmd = (
            f"NanoFilt -q {NANOFILT_QSCORE} "
            f"--length {NANOFILT_MIN_LEN} --maxlength {NANOFILT_MAX_LEN} "
            f"{src} > {output_path}"
        )
        self._exec(cmd)

        # Count raw and filtered reads
        raw_fastq_lines = sum(1 for _ in open(src, "r")) / 4
        filtered_fastq_lines = sum(1 for _ in open(output_path, "r")) / 4
        percentage = (filtered_fastq_lines / raw_fastq_lines * 100) if raw_fastq_lines else 0
        logger.info(
            f"Raw reads: {int(raw_fastq_lines)}, "
            f"Passed: {int(filtered_fastq_lines)} ({int(percentage)}%)"
        )
        return output_path

    def _average_quality(self, quality_string: str) -> float:
        """
        Calculate the average Phred quality score from a FASTQ quality string.
        """
        scores = [ord(char) - 33 for char in quality_string]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def qualityfilt(
        self,
        src: str,
        des: str,
        name: str = "all.fastq",
        QSCORE: int = 8,
        MIN_LEN: int = 400,
        MAX_LEN: int = 8000,
        SEARCH_SEQ: Optional[List[str]] = None,
        EDIT_DISTANCE: int = 0,
    ) -> str:
        """
        Filter FASTQ reads by average quality, length, and optional presence of a DNA subsequence.
        Returns the path to the filtered FASTQ file.
        """
        os.makedirs(des, exist_ok=True)
        logger.info("Starting qualityfilt...")
        logger.info(f"QSCORE: {QSCORE}, MIN_LEN: {MIN_LEN}, MAX_LEN: {MAX_LEN}")

        if SEARCH_SEQ:
            logger.info("SEARCH_SEQ is defined: searching for specified DNA sequences.")
            SEARCH_SEQ = [s.upper() for s in SEARCH_SEQ]
        else:
            SEARCH_SEQ = []

        output_path = os.path.join(des, name)
        total_reads = 0
        passed_reads = 0

        with open(src, "r") as infile, open(output_path, "w") as outfile:
            for rec in self._fastq_reader(infile):
                total_reads += 1
                seq_upper = rec["seq"].upper()
                rec["seq"] = seq_upper

                # Determine if sequence passes the SEARCH_SEQ criterion
                search_flag = True
                if SEARCH_SEQ:
                    search_flag = False
                    for pattern in SEARCH_SEQ:
                        edit_ctl = edlib.align(pattern, seq_upper, mode="HW", task="locations")
                        if edit_ctl["editDistance"] <= EDIT_DISTANCE:
                            search_flag = True
                            break

                # Check quality and length
                avg_q = self._average_quality(rec["qual"])
                seq_len = len(seq_upper)
                if (
                    avg_q >= QSCORE
                    and MIN_LEN <= seq_len <= MAX_LEN
                    and search_flag
                ):
                    passed_reads += 1
                    outfile.write(f"@{rec['title']}\n{seq_upper}\n+\n{rec['qual']}\n")

                if total_reads % 1000 == 0:
                    logger.debug(
                        f"{passed_reads}/{total_reads} "
                        f"({passed_reads / total_reads * 100:.2f}%) reads passed so far..."
                    )

        percent_final = (passed_reads / total_reads * 100) if total_reads else 0
        logger.info(
            f"Filtering complete: {passed_reads}/{total_reads} "
            f"({percent_final:.2f}%) reads passed the quality filter."
        )
        return output_path

    def minibar(
        self,
        src: str,
        des: str,
        BARCODE_INDEX_FILE: str,
        MINIBAR_INDEX_DIS: int,
    ) -> str:
        """
        Perform demultiplexing with minibar.py using the provided barcode index file
        and maximum edit distance. Returns the output directory path.
        """
        os.makedirs(des, exist_ok=True)
        logger.info("Checking barcode index file...")

        # Validate barcode index file by capturing stderr
        result = self._exec(f"minibar.py {BARCODE_INDEX_FILE} -info cols", suppress_output=False)
        if result is None:
            raise RuntimeError("Unable to validate barcode index file: no output captured.")
        _, stderr = result
        if stderr:
            error_msg = stderr.decode("utf-8", errors="ignore").strip()
            logger.error(f"Invalid barcode index file: {error_msg}")
            raise ValueError("Invalid barcode index file")
        else:
            logger.info(f"{BARCODE_INDEX_FILE} is valid.")

        cwd = os.getcwd()
        try:
            os.chdir(des)
            cmd = (
                f"minibar.py -F -C -e {MINIBAR_INDEX_DIS} "
                f"{BARCODE_INDEX_FILE} {src} 2>&1"
            )
            result = self._exec(cmd, suppress_output=False)
            if result is None:
                raise RuntimeError("Demultiplexing failed with no output.")
            _, stderr2 = result
            if stderr2:
                stderr_text = stderr2.decode("utf-8", errors="ignore").strip()
                logger.error(f"Minibar error: {stderr_text}")
                raise RuntimeError("Minibar demultiplexing failed.")
        finally:
            os.chdir(cwd)

        return des

    def _fastq_to_fasta(self, src: str, des: str) -> str:
        """
        Convert a single FASTQ file to FASTA format.
        Returns the path to the created FASTA file.
        """
        with open(src, "r") as infile, open(des, "w") as outfile:
            for rec in self._fastq_reader(infile, suppress_warning=True):
                outfile.write(f">{rec['title']}\n{rec['seq']}\n")
        logger.info(f"Converted {src} to {des}")
        return des

    def batch_to_fasta(self, src: str, des: str, ext: str = "fas") -> str:
        """
        Convert all FASTQ files in 'src' directory to FASTA files in 'des' directory.
        Returns the output directory path.
        """
        logger.info("Starting batch FASTQ to FASTA conversion...")
        os.makedirs(des, exist_ok=True)
        for entry in os.scandir(src):
            if entry.is_file() and entry.name.endswith(".fastq"):
                logger.info(f"Converting {entry.name}")
                sample_id, _ = os.path.splitext(entry.name)
                out_path = os.path.join(des, f"{sample_id}.{ext}")
                self._fastq_to_fasta(entry.path, out_path)
        return des

    def distance_matrix(self, path: str, TRUNCATE_HEAD_TAIL: bool = True) -> np.ndarray:
        """
        Compute a pairwise distance matrix for all sequences in a FASTA/FASTQ file.
        If TRUNCATE_HEAD_TAIL is True, attempts to truncate heads and tails of each sequence.
        Returns a symmetric numpy.ndarray of distances.
        """
        sample_id, extension = os.path.splitext(os.path.basename(path))
        ext = extension.lstrip(".")
        sequences: List[Dict[str, str]]
        with open(path, "r") as infile:
            if ext in self.fastq_ext:
                sequences = list(self._fastq_reader(infile))
            elif ext in self.fasta_ext:
                sequences = list(self._fasta_reader(infile))
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        if TRUNCATE_HEAD_TAIL:
            for seq_record in sequences:
                try:
                    seq_record["seq"] = self._truncate_head_and_tail(seq_record["seq"])
                except Exception:
                    # HEAD/Tail labels not found; leaving sequence unchanged
                    pass

        num_records = len(sequences)
        logger.info(f"Number of records: {num_records}")
        dm = np.zeros((num_records, num_records), dtype=float)

        for i in range(num_records):
            seq_i = sequences[i]["seq"]
            for j in range(i + 1, num_records):
                seq_j = sequences[j]["seq"]
                dist_fw = self._pairwise_distance(seq_i, seq_j)
                dist_rev = self._pairwise_distance(seq_i, self._reverse_complement(seq_j))
                d = min(dist_fw, dist_rev)
                dm[i, j] = d
                dm[j, i] = d

        return dm

    def _hdbscan(
        self,
        dm: np.ndarray,
        min_cluster_size: int = 6,
        min_samples: int = 1
    ) -> np.ndarray:
        """
        Perform HDBSCAN clustering on a precomputed distance matrix dm.
        Returns an array of cluster labels.
        """
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clusterer.fit(dm)
        return clusterer.labels_

    def hdbscan(
        self,
        src: str,
        des: str,
        input_format: str = "fastq",
        output_format: str = "both",
        min_cluster_size: float = 0.3,
        mds: bool = True
    ) -> None:
        """
        For each file in 'src' matching input_format, compute distance matrix,
        perform HDBSCAN clustering, and write cluster FASTA files to 'des'.
        If mds is True, also generate a scatter plot of clusters in two dimensions.
        """
        io_format = self._check_input_output(input_format=input_format, output_format=output_format)
        os.makedirs(des, exist_ok=True)
        abs_des = os.path.abspath(des)

        for entry in os.scandir(src):
            sample_id, extension = os.path.splitext(entry.name)
            ext = extension.lstrip(".")

            if not (entry.is_file() and ext == io_format["input"]):
                logger.info(f"Skipping {entry.name}: not a {io_format['input']} file.")
                continue

            clustered_seq: Dict[int, List[Dict[str, str]]] = {}
            logger.info(f"Clustering {entry.name}")

            try:
                dm = self.distance_matrix(entry.path)
            except Exception as e:
                logger.error(f"Failed to compute distance matrix for {entry.name}: {e}")
                continue

            abs_cluster_size = max(2, int(dm.shape[0] * min_cluster_size))
            logger.info(f"Computed absolute cluster size: {abs_cluster_size}")

            try:
                labels = self._hdbscan(dm, abs_cluster_size)
            except Exception as e:
                logger.error(f"Clustering failed for {entry.name}: {e}")
                continue

            # Read all sequences again to group by cluster label
            try:
                with open(entry.path, "r") as infile:
                    if ext in self.fastq_ext:
                        seqs = list(self._fastq_reader(infile))
                    else:
                        seqs = list(self._fasta_reader(infile))
            except Exception as e:
                logger.error(f"Failed to read sequences from {entry.name}: {e}")
                continue

            for idx, label in enumerate(labels):
                clustered_seq.setdefault(label, []).append(seqs[idx])

            logger.info(f"Number of clusters in {entry.name}: {len(clustered_seq)}")

            # Write clustered FASTA files
            for label, records in clustered_seq.items():
                out_file = os.path.join(abs_des, f"{sample_id}_cluster_{label}_r{len(records)}.fas")
                try:
                    with open(out_file, "w") as outfile:
                        for rec in records:
                            outfile.write(f">{rec['title']}\n{rec['seq']}\n")
                    logger.info(f"Wrote cluster {label} to {out_file}")
                except Exception as e:
                    logger.error(f"Failed to write {out_file}: {e}")

            # Optionally visualize with MDS
            if mds and dm.size > 0:
                try:
                    dm_norm = dm / (dm.max() + 0.01)
                    mds_model = MDS(
                        n_components=2,
                        random_state=5566,
                        dissimilarity="precomputed",
                        normalized_stress="auto"
                    )
                    coords = mds_model.fit_transform(dm_norm)
                    fig, ax = plt.subplots(figsize=(15, 15))
                    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="rainbow", s=18)
                    plot_path = os.path.join(des, f"{entry.name[:-len(extension)]}_MDS.jpg")
                    plt.savefig(plot_path, dpi=56)
                    plt.close(fig)
                    logger.info(f"MDS plot saved to {plot_path}")
                except Exception as e:
                    logger.error(f"Failed to generate MDS plot for {entry.name}: {e}")
    def lamassemble(
        self,
        src: str,
        des: str,
        mat: str = "/content/lamassemble/train/promethion.mat"
    ) -> str:
        """
        Run lamassemble on all .fas files in the src directory.
        Outputs alignment files to des directory.
        """
        os.makedirs(des, exist_ok=True)
        for entry in os.scandir(src):
            if not entry.name.endswith(".fas"):
                continue
            input_path = entry.path
            aln_output = os.path.join(des, f"aln_{entry.name}")
            con_output = os.path.join(des, f"con_{entry.name}")
            # Align sequences
            cmd_align = f"lamassemble {mat} -a {input_path} > {aln_output}"
            self._exec(cmd_align, suppress_output=True)
            # Generate consensus
            sample_name = entry.name[:-4]
            cmd_consensus = (
                f"lamassemble {mat} -c -n {sample_name} {aln_output} > {con_output}"
            )
            self._exec(cmd_consensus, suppress_output=True)
            logger.info(f"Processed {entry.name}: alignment -> {aln_output}, consensus -> {con_output}")
        return des

    def _trim_by_case(
        self,
        src: str,
        des: str,
        fw_offset: int = 0,
        rv_offset: int = 0,
        input_format: str = "fastq",
        output_format: str = "both"
    ) -> str:
        """
        Trim reads based on labeled HEAD/barcode/SEQ/barcode/TAIL structure.
        Only works on files where sequences are labeled as:
        HEAD(uppercase)+barcode(lowercase)+SEQ(uppercase)+barcode(lowercase)+TAIL(uppercase).
        """
        os.makedirs(des, exist_ok=True)
        io_format = self._check_input_output(input_format, output_format)

        for entry in os.scandir(src):
            if not entry.is_file():
                continue

            sample_id, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            if ext not in io_format["input"]:
                logger.info(f"Skipping {entry.name}: not in input format {io_format['input']}")
                continue

            logger.info(f"Trimming by case: {entry.name}")
            infile_path = entry.path

            # Open input and output files
            with open(infile_path, "r") as infile:
                seq_iter: Iterator[Dict[str, str]]
                if ext in self.fastq_ext:
                    seq_iter = self._fastq_reader(infile)
                else:  # ext in fasta_ext
                    seq_iter = self._fasta_reader(infile)

                outfile_fasta: Optional[TextIO] = None
                outfile_fastq: Optional[TextIO] = None
                if "fasta" in io_format["output"]:
                    fasta_ext = io_format["output"]["fasta"]
                    fasta_path = os.path.join(des, f"{sample_id}.{fasta_ext}")
                    outfile_fasta = open(fasta_path, "w")
                    logger.debug(f"Created FASTA output: {fasta_path}")
                if "fastq" in io_format["output"]:
                    fastq_ext = io_format["output"]["fastq"]
                    fastq_path = os.path.join(des, f"{sample_id}.{fastq_ext}")
                    outfile_fastq = open(fastq_path, "w")
                    logger.debug(f"Created FASTQ output: {fastq_path}")

                for record in seq_iter:
                    seq = record["seq"]
                    title = record["title"]
                    qual = record.get("qual", "")
                    try:
                        match = re.search(r"([A-Z]+)([a-z]+)([A-Z]+)([a-z]+)([A-Z]+)", seq)
                        if not match:
                            raise ValueError(f"No labeled HEAD found in {title}")

                        # Calculate trimmed sequence indices
                        start_idx = match.start(3) + fw_offset
                        end_idx = match.end(3) - rv_offset
                        trimmed_seq = seq[start_idx:end_idx]
                        if "qual" in record:
                            trimmed_qual = qual[start_idx:end_idx]
                        else:
                            trimmed_qual = ""

                        # Write to outputs
                        if outfile_fasta:
                            outfile_fasta.write(f">{title}\n{trimmed_seq}\n")
                        if outfile_fastq:
                            outfile_fastq.write(f"@{title}\n{trimmed_seq}\n+\n{trimmed_qual}\n")
                    except Exception as exc:
                        logger.debug(f"Skipping {title}: {exc}")

                if outfile_fasta:
                    outfile_fasta.close()
                if outfile_fastq:
                    outfile_fastq.close()

        return des

    def _trim_by_seq(
        self,
        src: str,
        des: str,
        BARCODE_INDEX_FILE: str,
        fw_col: str = "FwPrimer",
        rv_col: str = "RvPrimer",
        input_format: str = "fastq",
        output_format: str = "both",
        fw_offset: int = 0,
        rv_offset: int = 0,
        mismatch_ratio_f: float = 0.15,
        mismatch_ratio_r: float = 0.15,
        discard_no_match: bool = False,
        check_both_directions: bool = True,
        reverse_complement_rv_col: bool = True,
        search_range: int = 200
    ) -> str:
        """
        Trim reads by matching forward and reverse barcode sequences.
        BARCODE_INDEX_FILE must be a .tsv or .csv with columns: SampleID, fw_col, rv_col.
        """
        os.makedirs(des, exist_ok=True)
        io_format = self._check_input_output(input_format, output_format)

        # Load barcode index table
        try:
            # Load barcode index
            if BARCODE_INDEX_FILE.endswith(".tsv"):
                df = pd.read_csv(BARCODE_INDEX_FILE, sep="\t")
            elif BARCODE_INDEX_FILE.endswith(".csv"):
                df = pd.read_csv(BARCODE_INDEX_FILE)
            elif BARCODE_INDEX_FILE.endswith(".xlsx"):
                df = pd.read_excel(BARCODE_INDEX_FILE)
            else:
                logger.error(f"Unsupported barcode index file format: {BARCODE_INDEX_FILE}")
                
            if not all(col in df.columns for col in ["SampleID", fw_col, rv_col]):
                raise KeyError(f"BARCODE_INDEX_FILE must contain columns: SampleID, {fw_col}, {rv_col}")
            df = df.astype(str)
        except Exception as exc:
            logger.error(f"Error loading barcode index: {exc}")
            return des

        # Process each sample file
        for entry in os.scandir(src):
            if not entry.is_file():
                continue

            sample_id, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            if ext not in io_format["input"]:
                logger.info(f"Skipping {entry.name}: not in input format {io_format['input']}")
                continue

            logger.info(f"Trimming by sequence: {entry.name}")
            try:
                fw_trim = df.loc[df["SampleID"] == sample_id, fw_col].values[0].upper()
                rv_trim = df.loc[df["SampleID"] == sample_id, rv_col].values[0].upper()
                if reverse_complement_rv_col:
                    rv_trim = self._reverse_complement(rv_trim)
            except Exception as exc:
                logger.error(f"Barcode lookup failed for {sample_id}: {exc}")
                continue

            infile_path = entry.path
            with open(infile_path, "r") as infile:
                if ext in self.fastq_ext:
                    seq_iter = self._fastq_reader(infile)
                else:
                    seq_iter = self._fasta_reader(infile)

                # Prepare output files
                outfile_fasta: Optional[TextIO] = None
                outfile_fastq: Optional[TextIO] = None
                if "fasta" in io_format["output"]:
                    fasta_ext = io_format["output"]["fasta"]
                    fasta_path = os.path.join(des, f"{sample_id}.{fasta_ext}")
                    outfile_fasta = open(fasta_path, "w")
                    logger.debug(f"Created FASTA output: {fasta_path}")
                if "fastq" in io_format["output"]:
                    fastq_ext = io_format["output"]["fastq"]
                    fastq_path = os.path.join(des, f"{sample_id}.{fastq_ext}")
                    outfile_fastq = open(fastq_path, "w")
                    logger.debug(f"Created FASTQ output: {fastq_path}")

                trimmed_F = 0
                trimmed_R = 0
                total = 0

                for record in seq_iter:
                    total += 1
                    seq = record["seq"]
                    title = record["title"]
                    qual = record.get("qual", "")
                    seq_upper = seq.upper()

                    # Attempt forward trim
                    fw_align = edlib.align(
                        fw_trim,
                        seq_upper[:search_range],
                        mode="HW",
                        task="locations",
                        k=int(len(fw_trim) * mismatch_ratio_f)
                    )
                    if fw_align["locations"]:
                        trimmed_F += 1
                        idx_end = fw_align["locations"][0][1] + 1 + fw_offset
                        seq = seq[idx_end:]
                        if "qual" in record:
                            qual = qual[idx_end:]
                        seq_upper = seq.upper()

                    # Attempt reverse trim
                    rv_align = edlib.align(
                        rv_trim,
                        seq_upper[-search_range:],
                        mode="HW",
                        task="locations",
                        k=int(len(rv_trim) * mismatch_ratio_r)
                    )
                    if rv_align["locations"]:
                        trimmed_R += 1
                        splice_site = len(seq) - search_range + rv_align["locations"][0][0] - rv_offset
                        seq = seq[:splice_site]
                        if "qual" in record:
                            qual = qual[:splice_site]

                    # If neither found and checking both directions
                    if not fw_align["locations"] and not rv_align["locations"] and check_both_directions:
                        rc_seq = self._reverse_complement(seq)
                        rc_qual = qual[::-1] if "qual" in record else ""
                        fw_align_rc = edlib.align(
                            fw_trim,
                            rc_seq.upper()[:search_range],
                            mode="HW",
                            task="locations",
                            k=int(len(fw_trim) * mismatch_ratio_f)
                        )
                        if fw_align_rc["locations"]:
                            trimmed_F += 1
                            idx_end_rc = fw_align_rc["locations"][0][1] + 1 + fw_offset
                            rc_seq = rc_seq[idx_end_rc:]
                            if "qual" in record:
                                rc_qual = rc_qual[idx_end_rc:]
                        rv_align_rc = edlib.align(
                            rv_trim,
                            rc_seq.upper()[-search_range:],
                            mode="HW",
                            task="locations",
                            k=int(len(rv_trim) * mismatch_ratio_r)
                        )
                        if rv_align_rc["locations"]:
                            trimmed_R += 1
                            splice_site_rc = (
                                len(rc_seq) - search_range + rv_align_rc["locations"][0][0] - rv_offset
                            )
                            rc_seq = rc_seq[:splice_site_rc]
                            if "qual" in record:
                                rc_qual = rc_qual[:splice_site_rc]

                        # If still no match and discarding is requested
                        if not fw_align_rc["locations"] and not rv_align_rc["locations"] and discard_no_match:
                            continue

                        # Restore original orientation
                        seq = self._reverse_complement(rc_seq)
                        qual = rc_qual[::-1] if "qual" in record else ""

                    # Skip empty sequences
                    if not seq:
                        logger.debug(f"Discarded {title}: empty sequence after trimming")
                        continue

                    # Write trimmed record
                    if outfile_fasta:
                        outfile_fasta.write(f">{title}\n{seq}\n")
                    if outfile_fastq:
                        outfile_fastq.write(f"@{title}\n{seq}\n+\n{qual}\n")

                # Close output handles
                if outfile_fasta:
                    outfile_fasta.close()
                if outfile_fastq:
                    outfile_fastq.close()

                logger.info(
                    f"{sample_id}: Total reads={total}, trimmed forward={trimmed_F}, trimmed reverse={trimmed_R}"
                )

        return des

    def trim_reads(
        self,
        src: str,
        des: str,
        mode: str = "case",
        input_format: str = "fastq",
        output_format: str = "both",
        BARCODE_INDEX_FILE: str = "",
        fw_col: str = "FwPrimer",
        rv_col: str = "RvPrimer",
        fw_offset: int = 0,
        rv_offset: int = 0,
        mismatch_ratio_f: float = 0.15,
        mismatch_ratio_r: float = 0.15,
        discard_no_match: bool = False,
        check_both_directions: bool = True,
        reverse_complement_rv_col: bool = True,
        search_range: int = 200
    ) -> str:
        """
        Wrapper to trim reads by either 'case' or 'table' mode.
        'case': trim using labeled sequence patterns.
        'table': trim using a barcode index file.
        """
        self._check_input_output(input_format, output_format)
        if mode == "table":
            self._trim_by_seq(
                src=src,
                des=des,
                BARCODE_INDEX_FILE=BARCODE_INDEX_FILE,
                fw_col=fw_col,
                rv_col=rv_col,
                input_format=input_format,
                output_format=output_format,
                fw_offset=fw_offset,
                rv_offset=rv_offset,
                mismatch_ratio_f=mismatch_ratio_f,
                mismatch_ratio_r=mismatch_ratio_r,
                discard_no_match=discard_no_match,
                check_both_directions=check_both_directions,
                reverse_complement_rv_col=reverse_complement_rv_col,
                search_range=search_range,
            )
        elif mode == "case":
            logger.info(
                "Notice: mode is set to 'case'; other arguments will be ignored "
                "except src, des, fw_offset, rv_offset, input_format, output_format."
            )
            self._trim_by_case(
                src=src,
                des=des,
                fw_offset=fw_offset,
                rv_offset=rv_offset,
                input_format=input_format,
                output_format=output_format,
            )
        else:
            raise ValueError("mode must be either 'case' or 'table'")
        return des
    def blast(
        self,
        src: str,
        des: str,
        name: str = "blast.csv",
        funguild: bool = True,
        startswith: str = "con_"
    ) -> str:
        """
        For each .fas file in src that starts with 'startswith', perform BLAST on each sequence.
        Fetches fungal guild data if funguild=True. Outputs results to a CSV named 'name' in des.
        """
        os.makedirs(des, exist_ok=True)
        results: List[Dict[str, Any]] = []

        for entry in os.scandir(src):
            if not (entry.name.startswith(startswith) and entry.name.endswith(".fas")):
                continue

            logger.info(f"Blasting {entry.name}")
            row: Dict[str, Any] = {}

            # Read all sequences in this FASTA file
            with open(entry.path, "r") as handle:
                sequences = list(self._fasta_reader(handle))

            for s in sequences:
                row = {
                    "name": s["title"],
                    "seq": s["seq"],
                    "length": len(s["seq"])
                }
                try:
                    # Parse cluster and reads from title if present
                    info = s["title"].split("cluster", 1)[1]
                    parts = info.split("_")
                    row["cluster"] = parts[1]
                    row["reads"] = parts[2].lstrip("r")
                except Exception:
                    # If parsing fails, leave cluster/reads absent
                    pass

                # Perform BLAST
                try:
                    blast_res = self._blast(row["seq"])
                    row["organism"] = blast_res.get("org", "")
                    row["taxa"] = blast_res.get("taxa", "")
                    row["BLAST_simil"] = blast_res.get("sim", "")
                    row["BLAST_acc"] = blast_res.get("acc", "")
                    row["BLAST_seq"] = blast_res.get("seq", "")
                except Exception as exc:
                    logger.debug(f"BLAST failed for {row['name']}: {exc}")

                # Optionally fetch funguild information
                if funguild:
                    row["funguild"] = ""
                    row["funguild_notes"] = ""
                    try:
                        url = (
                            "https://www.mycoportal.org/funguild/services/api/"
                            f"db_return.php?qDB=funguild_db&qField=taxon&qText={urllib.parse.quote(row.get('organism', ''))}"
                        )
                        response = get(url, timeout=10)
                        data = response.json() if response.status_code == 200 else []
                        if data:
                            row["funguild"] = data[0].get("guild", "")
                            row["funguild_notes"] = data[0].get("notes", "")
                    except Exception as exc:
                        logger.debug(f"Funguild lookup failed for {row.get('organism', '')}: {exc}")

                logger.debug(f"Appending row: {row}")
                results.append(row)

        # Write all results to CSV
        output_path = os.path.join(des, name)
        try:
            pd.DataFrame(results).to_csv(output_path, index=False)
            logger.info(f"BLAST results written to {output_path}")
        except Exception as exc:
            logger.error(f"Failed to write BLAST CSV: {exc}")

        return output_path

    def _ensure_mmseqs_binary(self, option: str) -> str:
        """
        Ensure the specified MMseqs2 binary exists in the bin directory.
        If not present, download and extract the appropriate archive.
        option can be "linux_mmseqs", "linux_mmseqs_sse41", or "windows_mmseqs".
        Returns the full path to the executable.
        """
        bin_dir = os.path.join(self.lib_path, "bin")
        os.makedirs(bin_dir, exist_ok=True)

        # Determine download URL, archive name, and binary name based on option
        if option in ("linux_mmseqs", "linux_mmseqs_sse41"):
            url = "https://github.com/soedinglab/MMseqs2/releases/download/17-b804f/mmseqs-linux-sse41.tar.gz"
            archive_name = "mmseqs-linux-sse41.tar.gz"
            extract_type = "tar"
            binary_name = "mmseqs"
            expected_path = os.path.join(bin_dir, binary_name)
        elif option == "windows_mmseqs":
            url = "https://github.com/soedinglab/MMseqs2/releases/download/17-b804f/mmseqs-win64.zip"
            archive_name = "mmseqs-win64.zip"
            extract_type = "zip"
            # On Windows, executable is a .bat inside an "mmseqs" subfolder
            expected_path = os.path.join(bin_dir, "mmseqs", "mmseqs.bat")
        else:
            raise ValueError("mmseqs option must be 'linux_mmseqs', 'linux_mmseqs_sse41', or 'windows_mmseqs'")

        logger.debug(f"Expected executable path: {expected_path}")
        # If already present and executable, return immediately
        if os.path.isfile(expected_path) and os.access(expected_path, os.X_OK):
            logger.debug(f"Found existing MMseqs2 executable at: {expected_path}")
            return expected_path

        # Download archive
        tmp_archive = os.path.join(self.TEMP, archive_name)
        try:
            logger.info(f"Downloading MMseqs2 ({option}) from: {url}")
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(tmp_archive, "wb") as f_out:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f_out.write(chunk)
            logger.info(f"Download complete: {tmp_archive}")
        except Exception as exc:
            logger.error(f"Failed to download MMseqs2 archive: {exc}")
            raise

        # Extract the archive
        try:
            if extract_type == "tar":
                # Linux: extract only the 'mmseqs' binary
                with tarfile.open(tmp_archive, "r:gz") as tar:
                    for member in tar.getmembers():
                        name = os.path.basename(member.name)
                        if name == binary_name:
                            member.name = name  # avoid nested directories
                            tar.extract(member, bin_dir)
                            extracted_path = os.path.join(bin_dir, name)
                            os.chmod(extracted_path, 0o755)
                            logger.info(f"Extracted and set permissions: {extracted_path}")
                            break
                # Log bin_dir contents after extraction
                logger.debug(f"Contents of bin_dir after tar extraction: {os.listdir(bin_dir)}")

            else:  # ZIP for Windows
                with zipfile.ZipFile(tmp_archive, "r") as zf:
                    # Find top-level "mmseqs" folder in the archive
                    top_folder = None
                    for member in zf.namelist():
                        parts = member.split("/")
                        if parts and parts[0].lower() == "mmseqs":
                            top_folder = parts[0]
                            break
                    if not top_folder:
                        logger.error("Could not find a top-level 'mmseqs' folder in the ZIP archive.")
                        raise FileNotFoundError("ZIP archive does not contain an 'mmseqs' directory.")

                    # Extract entire "mmseqs" directory into bin_dir
                    for member in zf.namelist():
                        if member.startswith(f"{top_folder}/"):
                            rel_path = os.path.relpath(member, top_folder)
                            dest_path = os.path.join(bin_dir, "mmseqs", rel_path)
                            if member.endswith("/"):
                                os.makedirs(dest_path, exist_ok=True)
                                logger.debug(f"Created directory: {dest_path}")
                            else:
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                with zf.open(member) as source, open(dest_path, "wb") as target:
                                    shutil.copyfileobj(source, target)
                                # If this is the .bat, make it executable
                                if dest_path.lower().endswith(".bat"):
                                    os.chmod(dest_path, 0o755)
                                    logger.info(f"Extracted and set permissions: {dest_path}")
                    logger.info(f"Extracted 'mmseqs' folder to: {os.path.join(bin_dir, 'mmseqs')}")
                    # Log bin_dir contents after extraction
                    for root, dirs, files in os.walk(os.path.join(bin_dir, "mmseqs")):
                        for name in files:
                            file_path = os.path.join(root, name)
                            logger.debug(f"Found file: {file_path}")

        except Exception as exc:
            logger.error(f"Failed to extract MMseqs2 archive: {exc}")
            raise

        finally:
            # Clean up downloaded archive
            try:
                os.remove(tmp_archive)
            except OSError:
                logger.warning(f"Could not remove temporary archive: {tmp_archive}")

        # Final check: ensure the executable exists and is runnable
        if os.path.isfile(expected_path) and os.access(expected_path, os.X_OK):
            logger.debug(f"Binary is now present and executable: {expected_path}")
            return expected_path
        else:
            # If extraction did not yield the expected path, log directory state
            logger.error(f"Executable not found at expected path: {expected_path}")
            try:
                bin_contents = list(os.walk(bin_dir))
                logger.debug(f"Directory tree under '{bin_dir}': {bin_contents}")
            except Exception as e:
                logger.warning(f"Could not list bin_dir contents: {e}")

        raise FileNotFoundError(f"MMseqs2 executable not found or not executable at: {expected_path}")
    def mmseqs_cluster(
        self,
        src: str,
        des: str,
        mmseqs: str = "linux_mmseqs",       # "linux_mmseqs"、"linux_mmseqs_sse41" 或 "windows_mmseqs"
        input_format: str = "fastq",
        output_format: str = "both",
        min_seq_id: float = 0.5,
        cov_mode: int = 0,
        k: int = 14,
        threads: int = 8,
        s: float = 7.5,
        cluster_mode: Union[int, str] = 0,
        min_read_num: int = 0,
        min_read_ratio: float = 0.0,
        kmer_per_seq: int = 20,
        suppress_output: bool = True,
        tmp: str = "",
        dbtype: str = "2"
    ) -> str:
        """
        使用 MMseqs2 進行序列聚類：
        - mmseqs: 選擇 "linux_mmseqs"、"linux_mmseqs_sse41" 或 "windows_mmseqs" 來指定要使用或下載的二進位檔版本。
        """

        # 驗證 cluster_mode
        if cluster_mode not in [0, 1, 2, "linclust"]:
            raise ValueError("cluster_mode 必須是 0, 1, 2 或 'linclust'")

        # 取得並驗證 mmseqs 二進位檔
        mmseqs_bin = self._ensure_mmseqs_binary(mmseqs)

        # 驗證輸入與輸出格式
        io_format = self._check_input_output(input_format=input_format, output_format=output_format)
        os.makedirs(des, exist_ok=True)
        abs_des = os.path.abspath(des)

        for entry in os.scandir(src):
            if not entry.is_file():
                continue

            sample_id, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            if ext not in io_format["input"]:
                logger.info(f"跳過 {entry.name}: 不符合輸入格式 {io_format['input']}")
                continue

            # 清理 TEMP/資料夾
            self._clean_temp()
            # 設定臨時目錄
            if not tmp:
                tmp = os.path.join(self.TEMP, "tmp")
            os.makedirs(tmp, exist_ok=True)
            logger.info(f"處理樣本：{sample_id} ({entry.name})")
            # 若輸入為 FASTQ，先轉成 FASTA
            if ext in self.fasta_ext:
                fas_path = entry.path
            else:
                fas_path = os.path.join(tmp, "from_fastq.fas")
                self._fastq_to_fasta(entry.path, fas_path)

            logger.info(f"使用 MMseqs2 進行聚類：{entry.name}")
            db_path = os.path.join(tmp, "db")
            cluster_out = os.path.join(tmp, "cluster")

            # 建立 MMseqs2 資料庫
            cmd_create = f"{mmseqs_bin} createdb {fas_path} {db_path} --dbtype {dbtype}"
            self._exec(cmd_create, suppress_output=suppress_output)

            # 執行聚類
            if cluster_mode == "linclust":
                cmd_cluster = (
                    f"{mmseqs_bin} linclust {db_path} {cluster_out} {tmp} "
                    f"--kmer-per-seq {kmer_per_seq} --min-seq-id {min_seq_id} "
                    f"--cov-mode {cov_mode} --threads {threads}"
                )
            else:
                cmd_cluster = (
                    f"{mmseqs_bin} cluster {db_path} {cluster_out} {tmp} "
                    f"--min-seq-id {min_seq_id} --cov-mode {cov_mode} -k {k} "
                    f"--threads {threads} -s {s} --cluster-mode {cluster_mode}"
                )
            self._exec(cmd_cluster, suppress_output=suppress_output)

            # 輸出聚類結果
            seqdb_out = os.path.join(tmp, "cluster.seq")
            cmd_seqdb = f"{mmseqs_bin} createseqfiledb {db_path} {cluster_out} {seqdb_out}"
            self._exec(cmd_seqdb, suppress_output=suppress_output)
            cluster_fas_path = os.path.join(tmp, "cluster.fas")
            cmd_flat = (
                f"{mmseqs_bin} result2flat {db_path} {db_path} {seqdb_out} {cluster_fas_path}"
            )
            self._exec(cmd_flat, suppress_output=suppress_output)

            # 解析聚類 FASTA，並還原原始讀取
            bin_clusters: Dict[int, List[Dict[str, Any]]] = {}
            try:
                # 讀取原始序列
                with open(entry.path, "r") as rawfile:
                    if ext in self.fasta_ext:
                        raw_list = list(self._fasta_reader(rawfile))
                        raw_reads = {
                            r["title"]: {"title": r["title"], "seq": r["seq"]} 
                            for r in raw_list
                        }
                    else:
                        raw_list = list(self._fastq_reader(rawfile))
                        raw_reads = {
                            r["title"]: {"title": r["title"], "seq": r["seq"], "qual": r["qual"]} 
                            for r in raw_list
                        }
                    raw_reads_num = len(raw_reads)

                # 讀取聚類後的 FASTA
                with open(cluster_fas_path, "r") as cluster_handle:
                    cluster_no = -1
                    for rec in self._fasta_reader(cluster_handle):
                        if rec["seq"] == "":
                            cluster_no += 1
                            bin_clusters[cluster_no] = []
                        else:
                            title = rec["title"]
                            if title in raw_reads:
                                bin_clusters[cluster_no].append(raw_reads[title])
            except Exception as exc:
                logger.error(f"讀取聚類結果失敗 ({entry.name})：{exc}")
                continue

            logger.info(f"{sample_id}：聚類數量 = {len(bin_clusters)}")

            # 將符合門檻的 cluster 輸出
            for cno, members in bin_clusters.items():
                if len(members) < max(min_read_num, int(min_read_ratio * raw_reads_num)):
                    continue

                if "fasta" in io_format["output"]:
                    cluster_fas = os.path.join(
                        abs_des, f"{sample_id}_cluster_{cno}_r{len(members)}.fas"
                    )
                    with open(cluster_fas, "w") as h:
                        for rec in members:
                            h.write(f">{rec['title']}\n{rec['seq']}\n")

                if "fastq" in io_format["output"]:
                    cluster_fastq = os.path.join(
                        abs_des, f"{sample_id}_cluster_{cno}_r{len(members)}.fastq"
                    )
                    with open(cluster_fastq, "w") as h:
                        for rec in members:
                            h.write(f"@{rec['title']}\n{rec['seq']}\n+\n{rec['qual']}\n")

        return des


    def vsearch_OTUs(
        self,
        src: str,
        des: str,
        input_format: str = "fastq",
        output_format: str = "both",
        vsearch: str = "/nanoact/bin/vsearch", 
        id: float = 0.9,
        suppress_output: bool = True
    ) -> None:
        """
        Cluster sequences in src using VSEARCH at specified identity threshold.
        Outputs clusters (FASTA/FASTQ), OTU table, centroids, and .uc files to des.
        """
        lib_dir = os.path.dirname(os.path.realpath(__file__))
        vsearch_bin = os.path.join(lib_dir, "bin", "vsearch")

        io_format = self._check_input_output(input_format=input_format, output_format=output_format)
        os.makedirs(des, exist_ok=True)
        abs_des = os.path.abspath(des)

        for entry in os.scandir(src):
            sample_id, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            if not (entry.is_file() and ext == io_format["input"]):
                logger.info(f"Skipping {entry.name}: not in input format {io_format['input']}")
                continue

            logger.info(f"Clustering {entry.name} with VSEARCH")
            self._clean_temp()

            # Convert to FASTA if input is FASTQ
            if ext in self.fasta_ext:
                fas_path = entry.path
            else:
                fas_path = os.path.join(self.TEMP, "from_fastq.fas")
                self._fastq_to_fasta(entry.path, fas_path)

            # Run VSEARCH clustering
            uc_out = os.path.join(self.TEMP, "all.clustered.uc")
            otutab = os.path.join(self.TEMP, "all.otutab.txt")
            centroids = os.path.join(self.TEMP, "all.otus.fasta")
            clusters_dir = os.path.join(self.TEMP, "cluster")

            cmd_vsearch = (
                f"{vsearch_bin} --cluster_size {fas_path} --id {id} --strand plus "
                f"--sizein --sizeout --fasta_width 0 --uc {uc_out} "
                f"--relabel OTU_ --centroids {centroids} --otutabout {otutab} "
                f"--clusters {clusters_dir}"
            )
            self._exec(cmd_vsearch, suppress_output=suppress_output)

            # Read .uc file
            try:
                uc = pd.read_csv(uc_out, sep="\t", header=None)
            except Exception as exc:
                logger.error(f"Error reading UC file for {sample_id}: {exc}")
                continue

            # Filter and sort clusters
            uc = uc[uc[0].isin(["S", "H"])]
            uc.sort_values(by=8, ascending=False, inplace=True)

            # Read original sequences
            if input_format == "fastq":
                seqs = list(self._fastq_reader(open(entry.path, "r")))
            else:
                seqs = list(self._fasta_reader(open(entry.path, "r")))
            seqs.sort(key=lambda d: d["title"])

            # Map sequence titles to cluster IDs
            seq_cluster_pairs = uc[[8, 1]].to_dict(orient="records")
            clusters: Dict[str, List[Dict[str, Any]]] = {}

            for pair in seq_cluster_pairs:
                cluster_id = pair[1]
                prefix = pair[8]
                for seq_record in seqs:
                    if prefix in seq_record["title"]:
                        clusters.setdefault(cluster_id, []).append(seq_record)
                        seqs.remove(seq_record)
                        break

            # Write each cluster to file
            for cluster_label, members in clusters.items():
                fasta_path = os.path.join(abs_des, f"{sample_id}_cluster_{cluster_label}_r{len(members)}.fas")
                fastq_path = os.path.join(abs_des, f"{sample_id}_cluster_{cluster_label}_r{len(members)}.fastq")

                if "fasta" in io_format["output"]:
                    with open(fasta_path, "w") as h:
                        for rec in members:
                            h.write(f">{rec['title']}\n{rec['seq']}\n")

                if "fastq" in io_format["output"]:
                    with open(fastq_path, "w") as h:
                        for rec in members:
                            h.write(f"@{rec['title']}\n{rec['seq']}\n+\n{rec.get('qual', '')}\n")

            # Copy OTU outputs to destination
            try:
                shutil.copy(otutab, os.path.join(abs_des, f"{sample_id}_otu_table.txt"))
                shutil.copy(centroids, os.path.join(abs_des, f"{sample_id}.centroid"))
                shutil.copy(uc_out, os.path.join(abs_des, f"{sample_id}.uc"))
            except Exception as exc:
                logger.error(f"Failed to copy OTU output files for {sample_id}: {exc}")
    def cd_hit_est(
        self,
        src: str,
        des: str,
        input_format: str = "fastq",
        output_format: str = "both",
        cd_hit_est_path: str = "",
        identity_threshold: float = 0.8,
        word_size: int = 5,
        suppress_output: bool = True
    ) -> None:
        """
        Cluster sequences in src using CD-HIT-EST, writing clusters to des.
        
        Args:
            src: Directory containing input FASTA/FASTQ files.
            des: Directory to write clustered output files.
            input_format: Either "fasta" or "fastq". Only sequences of this format are processed.
            output_format: "fasta", "fastq", or "both".
            cd_hit_est_path: Path to the cd-hit-est executable. If empty, defaults to ./bin/cd-hit-est.
            identity_threshold: Sequence identity cutoff for clustering (0.0 to 1.0).
            word_size: Word size parameter (-n) for cd-hit-est.
            suppress_output: If True, suppress cd-hit-est stdout/stderr.
        """
        # Determine cd-hit-est binary location
        if not cd_hit_est_path:
            lib_dir = os.path.dirname(os.path.realpath(__file__))
            cd_hit_est_path = os.path.join(lib_dir, "bin", "cd-hit-est")

        self._check_input_output(input_format, output_format)
        os.makedirs(des, exist_ok=True)
        abs_des = os.path.abspath(des)

        for entry in os.scandir(src):
            if not entry.is_file():
                continue

            sample_id, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")

            # Skip files not matching the specified input_format
            if input_format == "fasta" and ext not in self.fasta_ext:
                continue
            if input_format == "fastq" and ext not in self.fastq_ext:
                continue

            logger.info(f"Clustering {entry.name}")
            self._clean_temp()

            # Convert FASTQ to FASTA if needed
            if ext in self.fastq_ext:
                fas_path = os.path.join(self.TEMP, "from_fastq.fas")
                self._fastq_to_fasta(entry.path, fas_path)
            else:
                fas_path = entry.path

            # Run cd-hit-est
            cdhit_output = os.path.join(self.TEMP, "cdhit.fas")
            cmd = (
                f"{cd_hit_est_path} -i {fas_path} -o {cdhit_output} "
                f"-c {identity_threshold} -n {word_size} -d 0"
            )
            self._exec(cmd, suppress_output=suppress_output)

            # Parse .clstr file
            clusters: Dict[str, List[str]] = {}
            clstr_path = f"{cdhit_output}.clstr"
            try:
                with open(clstr_path, "r") as clstr_handle:
                    current_cluster = ""
                    for line in clstr_handle:
                        if line.startswith(">Cluster"):
                            current_cluster = line.split()[1]
                            clusters[current_cluster] = []
                        else:
                            # Extract sequence title between '>' and '...'
                            parts = line.split(">")
                            if len(parts) > 1:
                                title = parts[1].split("...")[0].strip()
                                clusters[current_cluster].append(title)
            except Exception as exc:
                logger.error(f"Error reading cluster file {clstr_path}: {exc}")
                continue

            # Load original sequences into memory
            seq_list: List[Dict[str, str]]
            if input_format == "fastq":
                seq_list = list(self._fastq_reader(open(entry.path, "r")))
            else:
                seq_list = list(self._fasta_reader(open(entry.path, "r")))

            # Write clustered output
            for cluster_id, titles in clusters.items():
                cluster_size = len(titles)
                cluster_prefix = f"{sample_id}_cluster_{cluster_id}_r{cluster_size}"
                if "fastq" in output_format:
                    fastq_out_path = os.path.join(abs_des, f"{cluster_prefix}.fastq")
                    fastq_handle = open(fastq_out_path, "w")
                else:
                    fastq_handle = None

                if "fasta" in output_format:
                    fasta_out_path = os.path.join(abs_des, f"{cluster_prefix}.fas")
                    fasta_handle = open(fasta_out_path, "w")
                else:
                    fasta_handle = None

                for title in titles:
                    # Find and write matching record, then remove it from seq_list
                    for record in seq_list:
                        if title == record["title"]:
                            if fasta_handle:
                                fasta_handle.write(f">{record['title']}\n{record['seq']}\n")
                            if fastq_handle:
                                qual = record.get("qual", "")
                                fastq_handle.write(f"@{record['title']}\n{record['seq']}\n+\n{qual}\n")
                            seq_list.remove(record)
                            break

                if fasta_handle:
                    fasta_handle.close()
                if fastq_handle:
                    fastq_handle.close()

    def _calculate_5mer_frequency(self, sequence: str) -> Dict[str, int]:
        """
        Count occurrences of all 5-mers in the given sequence.
        """
        freq: Dict[str, int] = {}
        for i in range(len(sequence) - 4):
            kmer = sequence[i : i + 5]
            freq[kmer] = freq.get(kmer, 0) + 1
        return freq

    def fas_to_5mer(self, fas_path: str) -> pd.DataFrame:
        """
        Read a FASTA file and return a DataFrame of 5-mer frequencies per sequence.
        The DataFrame has a 'Sequence' column for identifiers, followed by one column per unique 5-mer.
        """
        freq_vectors: List[Dict[str, int]] = []
        seq_ids: List[str] = []

        with open(fas_path, "r") as handle:
            for rec in self._fasta_reader(handle):
                seq_id = rec["title"]
                sequence = rec["seq"]
                freq_vectors.append(self._calculate_5mer_frequency(sequence))
                seq_ids.append(seq_id)

        df = pd.DataFrame(freq_vectors).fillna(0).astype(int)
        df.insert(0, "Sequence", seq_ids)
        return df

    def random_sampler(
        self,
        src: str,
        des: str,
        input_format: str = "fastq",
        output_format: str = "both",
        ratio: float = 0.2
    ) -> None:
        """
        Randomly sample reads from a single FASTA/FASTQ file.
        Writes sampled subset to des directory, with filename indicating ratio.
        """
        io_format = self._check_input_output(input_format, output_format)
        os.makedirs(des, exist_ok=True)

        sample_id, ext = os.path.splitext(os.path.basename(src))
        ext = ext.lstrip(".")

        with open(src, "r") as handle:
            if ext in self.fasta_ext:
                seq_iter = self._fasta_reader(handle)
            elif ext in self.fastq_ext:
                seq_iter = self._fastq_reader(handle)
            else:
                logger.error(f"Unsupported file extension: {ext}")
                return

            # Prepare output handles if needed
            fasta_handle: Optional[TextIO] = None
            fastq_handle: Optional[TextIO] = None
            if "fasta" in io_format["output"]:
                outfasta = os.path.join(des, f"{sample_id}_{ratio}.{io_format['output']['fasta']}")
                fasta_handle = open(outfasta, "w")
            if "fastq" in io_format["output"]:
                outfastq = os.path.join(des, f"{sample_id}_{ratio}.{io_format['output']['fastq']}")
                fastq_handle = open(outfastq, "w")

            total = 0
            sampled = 0
            for record in seq_iter:
                total += 1
                if random() < ratio:
                    title = record["title"]
                    seq = record["seq"]
                    qual = record.get("qual", "")
                    if fasta_handle:
                        fasta_handle.write(f">{title}\n{seq}\n")
                    if fastq_handle:
                        fastq_handle.write(f"@{title}\n{seq}\n+\n{qual}\n")
                    sampled += 1

            if fasta_handle:
                fasta_handle.close()
            if fastq_handle:
                fastq_handle.close()

            self._p(f"Total reads: {total}, sampled reads: {sampled}, ratio: {sampled/total:.3f}")

    def region_extract(
        self,
        src: str,
        des: str,
        input_format: str = "fastq",
        output_format: str = "both",
        splicer: Dict[str, Tuple[str, ...]] = {
            "start": ("TCATTTAGAG", "GCCCGTCGCT", "GAAGTAAAAG", "TCGTAACAAG"),
            "end": ("GCTGAACTTA", "GCATATCAA", "ATCAATAAGCG", "AAGCGGAGGA")
        },
        k: int = 1
    ) -> None:
        """
        Extract a target region between start and end motifs (e.g., ITS region).
        Reads input files and writes extracted sub-sequences to des.
        """
        io_format = self._check_input_output(input_format, output_format)
        os.makedirs(des, exist_ok=True)

        for entry in os.scandir(src):
            sample_id, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")

            if ext != io_format["input"]:
                self._p(f"{entry.name} is not in the accepted input format, skipping")
                continue

            # Open input file
            if ext in self.fasta_ext:
                seq_iter = self._fasta_reader(open(entry.path, "r"))
            else:
                seq_iter = self._fastq_reader(open(entry.path, "r"))

            # Prepare output handles
            fasta_handle: Optional[TextIO] = None
            fastq_handle: Optional[TextIO] = None
            if "fasta" in io_format["output"]:
                fasta_out = os.path.join(des, f"{sample_id}.{io_format['output']['fasta']}")
                fasta_handle = open(fasta_out, "w")
            if "fastq" in io_format["output"]:
                fastq_out = os.path.join(des, f"{sample_id}.{io_format['output']['fastq']}")
                fastq_handle = open(fastq_out, "w")

            total_reads = 0
            total_extracted = 0

            for record in seq_iter:
                total_reads += 1
                seq = record["seq"]
                title = record["title"]
                qual = record.get("qual", "")
                seq_upper = seq.upper()

                start_idx = -1
                end_idx = -1

                # Find start motif
                for motif in splicer["start"]:
                    res = edlib.align(motif, seq_upper, mode="HW", task="locations", k=k)
                    if res["locations"]:
                        start_idx = res["locations"][0][0]
                        break

                # Find end motif (end of match)
                for motif in splicer["end"]:
                    res = edlib.align(motif, seq_upper, mode="HW", task="locations", k=k)
                    if res["locations"]:
                        end_idx = res["locations"][0][1] + 1
                        break

                # If valid region found
                if 0 <= start_idx < end_idx <= len(seq):
                    sub_seq = seq[start_idx:end_idx]
                    sub_qual = qual[start_idx:end_idx] if "qual" in record else ""
                    if fasta_handle:
                        fasta_handle.write(f">{title}\n{sub_seq}\n")
                    if fastq_handle:
                        fastq_handle.write(f"@{title}\n{sub_seq}\n+\n{sub_qual}\n")
                    total_extracted += 1

            if fasta_handle:
                fasta_handle.close()
            if fastq_handle:
                fastq_handle.close()

            self._p(
                f"{sample_id} Total reads: {total_reads}, "
                f"extracted reads: {total_extracted}, ratio: {total_extracted/total_reads:.3f}"
            )

    def _get_gbff_by_acc(self, accession_no: List[str]) -> str:
        """
        Fetch GBFF records from NCBI for given accession numbers.
        """
        ids = ",".join(accession_no)
        uri = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=nuccore&id={ids}&rettype=gbwithparts&retmode=text"
        )
        try:
            response = get(uri)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.error(f"Failed to fetch GBFF for {accession_no}: {exc}")
            return ""

    def _gbff_reader(self, handle: TextIO) -> Iterator[Dict[str, str]]:
        """
        Parse a GenBank flatfile (GBFF) from a file handle.
        Yields a dict with keys: accession, organism, taxid, seq.
        """
        data: Dict[str, str] = {}
        for line in handle:
            if line.startswith("ACCESSION"):
                data["accession"] = line.split()[1].strip()
            elif line.startswith("  ORGANISM"):
                data["organism"] = line.replace("  ORGANISM", "").strip()
            elif ' /db_xref="taxon:' in line:
                taxid = line.split('=')[1].replace('"', "").split(":")[1].strip()
                data["taxid"] = taxid
            elif line.startswith("ORIGIN"):
                seq_chars = []
                for seq_line in handle:
                    if seq_line.startswith("//"):
                        break
                    # Remove digits and whitespace
                    seq_line = re.sub(r"[\d\s]", "", seq_line)
                    seq_chars.append(seq_line.upper())
                data["seq"] = "".join(seq_chars)
                yield data
                data = {}
        # End of file
    def _lineage_by_taxid(self, taxid: List[str] = ["3016022", "2888342"]) -> Dict[str, Dict[str, str]]:
        """
        Retrieve lineage information (kingdom, phylum, class, order, family, genus) for each taxid.
        Caches results in self.tax_id_cache to avoid repeated NCBI calls.
        """
        # Load existing cache, if available
        try:
            with open(self.tax_id_cache, "r") as cache_fh:
                taxid_json: Dict[str, Dict[str, str]] = json.load(cache_fh)
        except Exception:
            taxid_json = {}

        ranks: Dict[str, Dict[str, str]] = {}
        to_query: List[str] = []

        # Separate cached vs. uncached taxids
        for t in taxid:
            if t in taxid_json:
                ranks[t] = taxid_json[t]
            else:
                to_query.append(t)

        # If all requested taxids are cached, return immediately
        if not to_query:
            return ranks

        # Fetch lineage info from NCBI for uncached taxids
        taxid_list_str = ",".join(to_query)
        taxid_info_URI = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=taxonomy&rettype=xml&id={taxid_list_str}"
        )
        logger.info(f"Fetching taxonomy for taxids: {taxid_list_str}")
        response = get(taxid_info_URI)
        try:
            xml_obj = xmltodict.parse(response.text)
        except Exception as e:
            logger.error(f"Failed to parse XML from NCBI for {taxid_info_URI}: {e}")
            return ranks

        # Prepare default rank placeholders
        rank_base = {
            "kingdom": "incertae sedis",
            "phylum": "incertae sedis",
            "class": "incertae sedis",
            "order": "incertae sedis",
            "family": "incertae sedis",
            "genus": "incertae sedis"
        }

        try:
            taxa = xml_obj["TaxaSet"]["Taxon"]
            # Ensure we have a list of Taxon entries
            if isinstance(taxa, dict):
                taxa = [taxa]

            for entry in taxa:
                this_rank = rank_base.copy()
                tid = entry["TaxId"]
                scientific_name = entry.get("ScientificName", "")
                primary_rank = entry.get("Rank", "")
                # Normalize superkingdom to kingdom
                if primary_rank == "superkingdom":
                    primary_rank = "kingdom"
                if primary_rank in this_rank:
                    this_rank[primary_rank] = scientific_name

                lineage_entries = entry.get("LineageEx", {}).get("Taxon", [])
                if isinstance(lineage_entries, dict):
                    lineage_entries = [lineage_entries]

                for lin in lineage_entries:
                    lin_rank = lin.get("Rank", "")
                    if lin_rank == "superkingdom":
                        lin_rank = "kingdom"
                    if lin_rank in this_rank:
                        this_rank[lin_rank] = lin.get("ScientificName", "")

                ranks[tid] = this_rank
                taxid_json[tid] = this_rank
                logger.debug(f"Cached lineage for taxid {tid}: {this_rank}")

        except Exception as e:
            logger.error(f"Error processing taxonomy XML for {taxid_list_str}: {e}")

        # Write updated cache back to disk
        try:
            os.makedirs(os.path.dirname(self.tax_id_cache), exist_ok=True)
            with open(self.tax_id_cache, "w") as cache_fh:
                json.dump(taxid_json, cache_fh)
        except Exception as e:
            logger.warning(f"Failed to write taxid cache: {e}")

        return ranks

    def _gbffgz_download(self, gbff_URI: str, des: str) -> str:
        """
        Download a .gbff.gz file from the given URI to the 'des' directory.
        If the downloaded file is gzipped, extract it and remove the .gz file.
        Returns the path to the extracted .gbff (or original if not gz).
        """
        os.makedirs(des, exist_ok=True)
        filename = os.path.basename(gbff_URI)
        basename, ext = os.path.splitext(filename)
        local_path = os.path.join(des, filename)

        logger.info(f"Downloading {filename} from NCBI RefSeq FTP...")
        try:
            resp = get(gbff_URI, allow_redirects=True)
            resp.raise_for_status()
            with open(local_path, "wb") as out_fh:
                out_fh.write(resp.content)
        except Exception as e:
            logger.error(f"Failed to download {gbff_URI}: {e}")
            return ""

        if filename.endswith(".gz"):
            extracted_path = os.path.join(des, basename)
            logger.info(f"Extracting {filename} to {extracted_path}...")
            try:
                with gzip.open(local_path, "rb") as gz_fh, open(extracted_path, "wb") as out_fh:
                    shutil.copyfileobj(gz_fh, out_fh)
                os.remove(local_path)
                return extracted_path
            except Exception as e:
                logger.error(f"Failed to extract {local_path}: {e}")
                return local_path
        else:
            return local_path

    def _gbffgz_to_taxfas(self, gbff_path: str, des: str) -> str:
        """
        Convert a .gbff (GenBank Flat File) to a taxonomy-annotated FASTA (.fas).
        Uses self._gbff_reader to parse each record and then queries lineage for each taxid.
        Returns the path to the generated .fas file.
        """
        os.makedirs(des, exist_ok=True)
        filename = os.path.basename(gbff_path)
        name_no_ext, _ = os.path.splitext(filename)
        recs = list(self._gbff_reader(open(gbff_path, "r")))

        logger.info(f"Gathering lineage info for {len(recs)} records...")
        taxinfos: Dict[str, Dict[str, str]] = {}
        taxid_set = {rec["taxid"] for rec in recs if "taxid" in rec}

        batch_size = 100
        taxid_list = list(taxid_set)
        for i in range(0, len(taxid_list), batch_size):
            slice_ids = taxid_list[i : i + batch_size]
            batch_info = self._lineage_by_taxid(slice_ids)
            taxinfos.update(batch_info)
            logger.debug(f"Processed {len(taxinfos)}/{len(taxid_set)} taxids")
        logger.info(f"Lineage retrieval complete: {len(taxinfos)}/{len(taxid_set)} taxids")

        output_fas = os.path.join(des, f"{name_no_ext}.fas")
        with open(output_fas, "w") as out_fh:
            for rec in recs:
                tid = rec.get("taxid", "")
                seq = rec.get("seq", "")
                lineage_parts = []
                try:
                    lineage_dict = taxinfos.get(tid, {})
                    lineage_parts = [
                        lineage_dict.get(rank, "Unclassified")
                        for rank in ["kingdom", "phylum", "class", "order", "family", "genus"]
                    ]
                except Exception:
                    lineage_parts = ["Unclassified"] * 6

                lineage_str = ";".join(lineage_parts)
                accession = rec.get("accession", "")
                organism = rec.get("organism", "")
                title = f"{accession}||{organism}||{lineage_str}||{tid}".replace(" ", "_")
                out_fh.write(f">{title}\n{seq}\n")

        logger.info(f"Generated taxonomy FASTA at: {output_fas}")
        return output_fas

    def _prepare_ref_db(
        self,
        des: str = "",
        custom_acc: List[str] = [],
        custom_gbff: List[str] = [],
        ref_db: List[str] = ["fungi.ITS", "bacteria.16SrRNA"]
    ) -> str:
        """
        Prepare a reference database by merging:
          - Custom accession numbers (custom_acc)
          - Custom .gbff URIs (custom_gbff)
          - Prebuilt ref_db entries from GitHub
        Outputs a single .fas file at 'des' (or self.TEMP/ref_db.fas if not provided).
        """
        if not des:
            des = os.path.join(self.TEMP, "ref_db.fas")

        self._clean_temp()
        custom_fas: List[str] = []

        # 1. Handle custom accession numbers: download and convert to FASTA
        if custom_acc:
            logger.info("Downloading custom accession records from NCBI...")
            custom_gbff_path = os.path.join(self.TEMP, "custom_db.gbff")
            all_gbff_text = ""
            for i in range(0, len(custom_acc), 100):
                batch_acc = custom_acc[i : i + 100]
                all_gbff_text += self._get_gbff_by_acc(batch_acc)
            with open(custom_gbff_path, "w") as out_fh:
                out_fh.write(all_gbff_text)
            custom_fas_path = self._gbffgz_to_taxfas(custom_gbff_path, self.TEMP)
            if custom_fas_path:
                custom_fas.append(custom_fas_path)

        # 2. Handle custom .gbff URIs
        if custom_gbff:
            logger.info("Downloading custom .gbff.gz URIs from NCBI...")
            for uri in custom_gbff:
                gbff_path = self._gbffgz_download(uri, self.TEMP)
                if gbff_path:
                    fas_path = self._gbffgz_to_taxfas(gbff_path, self.TEMP)
                    if fas_path:
                        custom_fas.append(fas_path)

        # 3. Merge all FASTA sources into a single reference FASTA
        logger.info("Merging custom databases and predefined ref_db entries...")
        os.makedirs(os.path.dirname(des), exist_ok=True)
        with open(des, "w") as combined_fh:
            # Download and append each predefined ref_db entry (gzip-compressed .fas)
            for entry in ref_db:
                try:
                    logger.info(f"Fetching ref_db entry: {entry}")
                    remote_uri = (
                        "https://github.com/Raingel/nanoact_refdb/raw/master/refdb/"
                        f"{entry}.fas.gz"
                    )
                    local_gz_path = os.path.join(self.TEMP, f"{entry}.fas.gz")
                    if not os.path.isfile(local_gz_path):
                        resp = get(remote_uri, allow_redirects=True)
                        resp.raise_for_status()
                        with open(local_gz_path, "wb") as gz_out:
                            gz_out.write(resp.content)
                    with gzip.open(local_gz_path, "rb") as gz_in:
                        combined_fh.write(gz_in.read().decode("utf-8"))
                    logger.debug(f"Appended {entry}.fas.gz to combined database")
                except Exception as e:
                    logger.warning(f"Failed to load {entry}.fas.gz: {e}")

            # Append custom FASTA files
            for fas_file in custom_fas:
                try:
                    with open(fas_file, "r") as cf:
                        combined_fh.write(cf.read())
                    logger.debug(f"Appended custom FASTA: {fas_file}")
                except Exception as e:
                    logger.warning(f"Failed to append custom FASTA {fas_file}: {e}")

        logger.info(f"Reference database prepared at: {des}")
        return des
    def local_blast(
        self,
        src: str,
        des: str,
        name: str = "blast.csv",
        startswith: str = "con_",
        input_format: str = "fastq",
        ref_db: List[str] = ["fungi.ITS", "bacteria.16SrRNA"],
        custom_acc: List[str] = [],
        custom_gbff: List[str] = [],
        suppress_mmseqs_output: bool = True,
    ) -> Optional[str]:
        """
        Perform local BLAST (via MMseqs2 easy-search) on all files in `src` that
        start with `startswith` and have extension `input_format`. Writes a single
        CSV report named `name` into `des`. Returns the path to the CSV or None on failure.
        """
        des = os.path.abspath(des)
        self._clean_temp()
        os.makedirs(des, exist_ok=True)

        # Prepare reference database (returns path to a single FASTA file or DB)
        ref_db_path = self._prepare_ref_db(
            des="",
            custom_acc=custom_acc,
            custom_gbff=custom_gbff,
            ref_db=ref_db,
        )

        # Merge all query sequences into one FASTA
        logger.info("Preparing query file...")
        query_path = os.path.join(self.TEMP, "query.fas")
        query_tmp: Dict[str, str] = {}
        file_count = 0
        rec_count = 0
        with open(query_path, "w") as query_handle:
            for entry in os.scandir(src):
                sample_id, ext = os.path.splitext(entry.name)
                ext = ext.lstrip(".")
                if ext != input_format or not entry.name.startswith(startswith):
                    continue

                with open(entry.path, "r") as handle:
                    if ext in self.fastq_ext:
                        recs = self._fastq_reader(handle)
                    elif ext in self.fasta_ext:
                        recs = self._fasta_reader(handle)
                    else:
                        logger.error(f"Unsupported input format: {ext}")
                        return None

                    for rec in recs:
                        title = rec["title"]
                        seq = rec["seq"]
                        query_handle.write(f">{title}\n{seq}\n")
                        query_tmp[title] = seq
                        rec_count += 1

                file_count += 1

        logger.info(f"Query file prepared: {rec_count} reads from {file_count} files.")
        if rec_count == 0:
            logger.warning("No reads in query file. Process terminated.")
            return None

        # Run MMseqs2 easy-search
        mmseqs_bin = os.path.join(self.lib_path, "bin", "mmseqs")
        result_m8 = os.path.join(self.TEMP, "result.m8")
        tmp_dir = os.path.join(self.TEMP, "tmp")
        cmd_easy_search = (
            f"{mmseqs_bin} easy-search {query_path} {ref_db_path} {result_m8} {tmp_dir} --search-type 3"
        )
        logger.info("Running mmseqs easy-search...")
        self._exec(cmd_easy_search, suppress_output=suppress_mmseqs_output)

        # Post-process result.m8
        logger.info("Post processing result.m8...")
        try:
            res_m8 = pd.read_csv(result_m8, sep="\t", header=None)
            res_m8.columns = [
                "qseqid",
                "sseqid",
                "pident",
                "length",
                "mismatch",
                "gapopen",
                "qstart",
                "qend",
                "sstart",
                "send",
                "evalue",
                "bitscore",
            ]
            # Keep, for each query, only the hit with the highest bitscore
            res_m8 = (
                res_m8.sort_values(by=["qseqid", "bitscore"], ascending=[True, False])
                .drop_duplicates(subset=["qseqid"], keep="first")
            )
        except Exception as exc:
            logger.error(
                f"Post processing failed. Turn off suppress_mmseqs_output and check MMseqs output. Error: {exc}"
            )
            return None

        # Build output DataFrame in one pass
        output_records: List[Dict[str, Union[str, int, float]]] = []
        for _, row in res_m8.iterrows():
            rec: Dict[str, Union[str, int, float]] = {}
            qid = row["qseqid"]
            seq = query_tmp.get(qid, "")
            rec["title"] = qid
            rec["seq"] = seq
            rec["fasta"] = f">{qid}\n{seq}\n"
            rec["length"] = len(seq)

            # Parse sample, cluster_no, reads_count from qseqid if matches pattern
            sample = cluster_no = ""
            reads_count = ""
            m = re.search(r"(.*)_cluster_([-0-9]+)_r(\d+)", qid)
            if m:
                sample, cluster_no, read_no = m.groups()
                reads_count = read_no.split(".")[0]
            rec["sample"] = sample
            rec["cluster_no"] = cluster_no
            rec["reads_count"] = reads_count

            # Parse sseqid into acc, hit_def, lineage, taxid
            acc = hit_def = lineage = taxid = ""
            kingdom = phylum = class_ = order = family = genus = ""
            try:
                acc, hit_def, lineage, taxid = row["sseqid"].split("||")
                kingdom, phylum, class_, order, family, genus = lineage.split(";")
            except Exception:
                pass

            rec["acc"] = acc
            rec["hit_def"] = hit_def
            rec["similarity"] = row["pident"]
            rec["org"] = hit_def
            rec["taxid"] = taxid
            rec["kingdom"] = kingdom
            rec["phylum"] = phylum
            rec["class"] = class_
            rec["order"] = order
            rec["family"] = family
            rec["genus"] = genus

            # Add funguild annotation
            guild, notes = self._funguild(hit_def)
            rec["funguild"] = guild
            rec["funguild_notes"] = notes

            output_records.append(rec)

        # Write final CSV
        blast_csv = os.path.join(des, name)
        output_df = pd.DataFrame(output_records)
        output_df.to_csv(blast_csv, index=False)
        logger.info(f"BLAST results saved to {blast_csv}")
        return blast_csv

    def taxonomy_assign(
        self,
        src: str,
        des: str,
        input_format: str = "fastq",
        lca_mode: int = 3,
        custom_acc: List[str] = [],
        custom_gbff: List[str] = [],
        ref_db: List[str] = ["fungi.ITS", "bacteria.16SrRNA"],
        sensitivity: float = 7.5,
        mmseqs: str = "linux_mmseqs",       # "linux_mmseqs"、"linux_mmseqs_sse41" 或 "windows_mmseqs"
        suppress_mmseqs_output: bool = True,
    ) -> None:
        """
        Assign taxonomy to sequences in `src` using MMseqs2 LCA-based taxonomy.
        Outputs TSV and HTML reports into `des` directory.
        """
        des = os.path.abspath(des)
        mode = "lca"
        self._clean_temp()
        os.makedirs(des, exist_ok=True)

        # Prepare reference database
        ref_db_path = self._prepare_ref_db(
            des="",
            custom_acc=custom_acc,
            custom_gbff=custom_gbff,
            ref_db=ref_db,
        )
        
        # 取得並驗證 mmseqs 二進位檔
        mmseqs_bin = self._ensure_mmseqs_binary(mmseqs)

        if mode == "lca":
            # Build reference MMseqs2 DB
            logger.info("Building reference DB from ref_db.fas...")
            self._exec(
                f"{mmseqs_bin} createdb {ref_db_path} {self.TEMP}/ref_db",
                suppress_output=suppress_mmseqs_output,
            )

            # Prepare taxid mapping file
            lookup_path = os.path.join(self.TEMP, "ref_db.lookup")
            taxidmap_path = os.path.join(self.TEMP, "ref_db.taxidmapping")
            with open(lookup_path, "r") as fh, open(taxidmap_path, "w") as out_fh:
                for line in fh:
                    parts = line.split("\t")
                    tax_id = parts[1].split("||")[-1]
                    parts[2] = tax_id
                    del parts[0]
                    out_fh.write("\t".join(parts))

            # Download and extract NCBI taxdump
            taxdump_uri = "https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz"
            taxdump_dir = os.path.join(self.TEMP, "ncbi-taxdump")
            os.makedirs(taxdump_dir, exist_ok=True)

            logger.info("Downloading taxdump from NCBI...")
            taxdump_archive = os.path.join(taxdump_dir, "taxdump.tar.gz")
            response = get(taxdump_uri)
            with open(taxdump_archive, "wb") as f:
                f.write(response.content)

            logger.info("Extracting taxdump...")
            with tarfile.open(taxdump_archive, "r:gz") as tar:
                tar.extractall(path=taxdump_dir)

            # Create taxonomy DB for MMseqs2
            logger.info("Creating taxonomy DB for MMseqs2...")
            self._exec(
                f"{mmseqs_bin} createtaxdb {self.TEMP}/ref_db {self.TEMP}/tmp "
                f"--ncbi-tax-dump {taxdump_dir} --tax-mapping-file {taxidmap_path}",
                suppress_output=suppress_mmseqs_output,
            )

        # Process each input file for taxonomy assignment
        input_exts = self._check_input_output(input_format, "fasta")["input"]
        for entry in os.scandir(src):
            sample_id, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".")
            if ext != input_exts:
                continue

            logger.info(f"Processing file: {entry.name}")
            query_fasta = os.path.join(self.TEMP, "query.fas")
            with open(query_fasta, "w") as fq:
                reader = (
                    self._fasta_reader(open(entry.path, "r"))
                    if ext in self.fasta_ext
                    else self._fastq_reader(open(entry.path, "r"))
                )
                for rec in reader:
                    fq.write(f">{rec['title']}\n{rec['seq']}\n")

            # Create MMseqs2 DB for query
            query_db = os.path.join(self.TEMP, f"{sample_id}_query_db")
            self._exec(
                f"{mmseqs_bin} createdb {query_fasta} {query_db}",
                suppress_output=suppress_mmseqs_output,
            )

            # Run LCA-based taxonomy
            tax_result = os.path.join(des, f"{sample_id}_taxonomyResult")
            self._exec(
                f"{mmseqs_bin} taxonomy {query_db} {self.TEMP}/ref_db {tax_result} "
                f"{self.TEMP}/tmp --search-type 3 --lca-mode {lca_mode} -s {sensitivity}",
                suppress_output=suppress_mmseqs_output,
            )

            # Generate TSV from taxonomy results
            tax_tsv = f"{tax_result}.tsv"
            self._exec(
                f"{mmseqs_bin} createtsv {query_db} {tax_result} {tax_tsv}",
                suppress_output=suppress_mmseqs_output,
            )

            # Generate taxonomy reports (plain and HTML)
            report_base = f"{sample_id}_taxonomyResultReport"
            report_txt = os.path.join(des, report_base)
            report_html = f"{report_txt}.html"
            self._exec(
                f"{mmseqs_bin} taxonomyreport {self.TEMP}/ref_db {tax_result} {report_txt}",
                suppress_output=suppress_mmseqs_output,
            )
            self._exec(
                f"{mmseqs_bin} taxonomyreport {self.TEMP}/ref_db {tax_result} {report_html} --report-mode 1",
                suppress_output=suppress_mmseqs_output,
            )

            logger.info(f"Taxonomy assignment completed for {sample_id}: {tax_tsv}, {report_txt}, {report_html}")
    def custom_taxonomy_sankey(
        self,
        src: str,
        des: str,
        img_ext: str = "png",
        minimal_reads: int = 1,
        vertical_scale: float = 1.0
    ) -> str:
        """
        Generate Sankey diagrams from taxonomy result TSV files in `src` and save images to `des`.
        Only include flows/nodes with counts >= minimal_reads. The plot height is scaled by `vertical_scale`.
        """
        from sankeyflow import Sankey

        os.makedirs(des, exist_ok=True)

        for entry in os.scandir(src):
            if not entry.name.endswith("_taxonomyResult.tsv"):
                continue

            result_tsv = entry.path
            sample_id, _ = os.path.splitext(entry.name)
            logger.info(f"Processing taxonomy file: {entry.name}")

            # Read raw TSV
            result_df_raw = pd.read_csv(result_tsv, sep="\t", header=None)

            # Count per taxid (column index 1)
            result_df = (
                result_df_raw.groupby(result_df_raw.columns[1])
                .count()
                .reset_index()
            )

            # Handle unclassified (taxid == 0)
            if 0 in result_df[1].tolist():
                unclassified_count = int(
                    result_df[result_df[1] == 0][0].sum()
                )
            else:
                unclassified_count = 0
            result_df = result_df[result_df[1] != 0]

            # Map tax_id to its assigned rank (column index 2)
            tax_id_rank = (
                result_df_raw[[1, 2]]
                .drop_duplicates()
                .set_index(1)[2]
                .to_dict()
            )

            # Prepare lists of tax IDs and counts
            tax_id_list = result_df[1].astype(str).tolist()
            tax_id_counts = result_df[0].tolist()

            # Query lineage for each tax_id in batches of 100
            tax_id_lineage: Dict[str, Dict[str, str]] = {}
            for i in range(0, len(tax_id_list), 100):
                batch = tax_id_list[i : i + 100]
                tax_id_lineage.update(self._lineage_by_taxid(batch))

            RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
            RANK_COLORS = [
                (255 / 255, 183 / 255, 178 / 255, 0.5),
                (205 / 255, 220 / 255, 57 / 255, 0.5),
                (100 / 255, 181 / 255, 246 / 255, 0.5),
                (255 / 255, 241 / 255, 118 / 255, 0.5),
                (255 / 255, 138 / 255, 101 / 255, 0.5),
                (171 / 255, 71 / 255, 188 / 255, 0.5),
            ]

            # Annotate raw DataFrame with lineage columns
            lineage_df = result_df_raw[1].astype(str).apply(
                lambda x: pd.Series(tax_id_lineage.get(x, {r: "" for r in RANKS}))
            )
            lineage_df.columns = RANKS
            result_df_raw = pd.concat([result_df_raw, lineage_df], axis=1)

            # Add frequency columns for each rank
            for r in RANKS:
                freq_col = f"{r}_freq"
                result_df_raw[freq_col] = result_df_raw[r].map(
                    result_df_raw[r].value_counts()
                )

            # Sort by the frequency of each rank (descending)
            sort_cols = [f"{r}_freq" for r in RANKS]
            result_df_raw = result_df_raw.sort_values(
                by=sort_cols, ascending=False
            )

            flow_pairs: List[Tuple[str, str]] = []
            flow_counts: List[int] = []
            node_labels: List[str] = []
            node_counts: List[int] = []

            # Build flow and node lists
            for _, row in result_df_raw.iterrows():
                identified_rank = row[2]
                for idx, r in enumerate(RANKS[:-1]):
                    if r == identified_rank:
                        break

                    next_rank = RANKS[idx + 1]
                    src_taxon = row[r]
                    dst_taxon = row[next_rank]
                    if not dst_taxon:
                        continue

                    src_label = f"{r}_{src_taxon}"
                    dst_label = f"{next_rank}_{dst_taxon}"
                    pair = (src_label, dst_label)

                    # Update flow
                    if pair not in flow_pairs:
                        flow_pairs.append(pair)
                        flow_counts.append(1)
                    else:
                        pos = flow_pairs.index(pair)
                        flow_counts[pos] += 1

                    # Update source node
                    if src_label not in node_labels:
                        node_labels.append(src_label)
                        node_counts.append(1)
                    else:
                        pos = node_labels.index(src_label)
                        node_counts[pos] += 1

                    # Update destination node if it's last step or identified_rank
                    if idx + 2 == len(RANKS) or next_rank == identified_rank:
                        if dst_label not in node_labels:
                            node_labels.append(dst_label)
                            node_counts.append(1)
                        else:
                            pos = node_labels.index(dst_label)
                            node_counts[pos] += 1

            # Prepare Sankey flows with colors based on source rank
            sankey_flows = [
                (
                    src,
                    dst,
                    count,
                    {"color": RANK_COLORS[RANKS.index(src.split("_")[0])]},
                )
                for (src, dst), count in zip(flow_pairs, flow_counts)
            ]

            # Organize nodes by rank
            sankey_nodes: List[List[Tuple[str, int, Dict[str, Any]]]] = [
                [] for _ in RANKS
            ]
            for label, count in zip(node_labels, node_counts):
                rank_key = label.split("_")[0]
                rank_index = RANKS.index(rank_key)
                sankey_nodes[rank_index].append((label, count, {"color": "grey"}))

            # Filter out flows and nodes below minimal_reads
            sankey_flows = [f for f in sankey_flows if f[2] >= minimal_reads]
            filtered_nodes = [
                [(lbl, cnt, opts) for (lbl, cnt, opts) in nodes if cnt >= minimal_reads]
                for nodes in sankey_nodes
            ]

            # Determine figure size based on genus-level node count
            genus_node_count = len(filtered_nodes[-1])
            fig_height = 0.8 * genus_node_count * vertical_scale
            fig, ax = plt.subplots(figsize=(30, fig_height))

            sankey_plot = Sankey(
                flows=sankey_flows,
                nodes=filtered_nodes,
                flow_color_mode="source",
                node_opts={
                    "label_format": "{label} ({value:.0f})",
                    "label_opts": {"fontsize": 16},
                    "label_pos": "right",
                },
                label_pad_x=10,
                scale=0.1,
            )
            sankey_plot.draw(ax=ax)
            ax.axis("off")

            output_path = os.path.join(des, f"{sample_id}.{img_ext}")
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Sankey saved: {output_path}")

        return des

    # Deprecated: use custom_taxonomy_sankey instead
    def _taxonomy_assign_visualizer(
        self,
        src: str,
        des: str,
        minimal_reads: int = 1,
        vertical_scale: float = 0.8
    ) -> None:
        """
        (Deprecated) Visualize taxonomy assignment via Sankey. 
        Uses `_taxonomyassignment.csv` files in `src` and saves results to `des`.
        """
        from sankeyflow import Sankey

        os.makedirs(des, exist_ok=True)

        for entry in os.scandir(src):
            if not entry.name.endswith("_taxonomyassignment.csv"):
                continue

            tax_assign = pd.read_csv(entry.path)
            sample_id = entry.name.replace("_taxonomyassignment.csv", "")
            logger.info(f"Processing taxonomy assignment: {entry.name}")

            RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
            RANK_COLORS = [
                (255 / 255, 183 / 255, 178 / 255, 0.5),
                (205 / 255, 220 / 255, 57 / 255, 0.5),
                (100 / 255, 181 / 255, 246 / 255, 0.5),
                (255 / 255, 241 / 255, 118 / 255, 0.5),
                (255 / 255, 138 / 255, 101 / 255, 0.5),
                (171 / 255, 71 / 255, 188 / 255, 0.5),
            ]

            # Add frequency columns for each rank
            for r in RANKS:
                freq_col = f"{r}_freq"
                tax_assign[freq_col] = tax_assign[r].map(tax_assign[r].value_counts())

            # Filter out low-frequency entries
            for r in RANKS:
                tax_assign = tax_assign[tax_assign[f"{r}_freq"] >= minimal_reads]

            # Sort by frequency descending for each rank
            sort_cols = [f"{r}_freq" for r in RANKS]
            tax_assign = tax_assign.sort_values(by=sort_cols, ascending=False)

            # Build source-target occurrence mapping
            src_tar: Dict[Tuple[int, int], int] = {}
            tax_list: List[str] = []
            tax_occurrence: Dict[str, int] = {}

            for _, row in tax_assign.iterrows():
                for idx, rank1 in enumerate(RANKS):
                    rank2 = RANKS[idx + 1] if idx + 1 < len(RANKS) else None
                    label1 = f"{rank1}_{row[rank1]}"
                    if label1 not in tax_list:
                        tax_list.append(label1)
                    tax_occurrence[label1] = tax_occurrence.get(label1, 0) + 1

                    if rank2 is None:
                        continue

                    label2 = f"{rank2}_{row[rank2]}"
                    if label2 not in tax_list:
                        tax_list.append(label2)

                    key = (tax_list.index(label1), tax_list.index(label2))
                    src_tar[key] = src_tar.get(key, 0) + 1

            # Build node structure and flow list for Sankey
            nodes_by_rank: Dict[str, List[str]] = {r: [] for r in RANKS}
            nodes_count: Dict[str, List[int]] = {r: [] for r in RANKS}
            flows: List[List[Any]] = []
            src_tar_records: List[Dict[str, Any]] = []

            for (i1, i2), count in src_tar.items():
                if count < minimal_reads:
                    continue
                src_label = tax_list[i1]
                dst_label = tax_list[i2]
                rank_source = src_label.split("_")[0]
                color = RANK_COLORS[RANKS.index(rank_source)]
                flows.append([src_label, dst_label, count, {"color": color}])

                src_tar_records.append(
                    {
                        "source": src_label,
                        "target": dst_label,
                        "source_id": i1,
                        "target_id": i2,
                        "value": count,
                        "source_rank": rank_source,
                        "color_by_rank": color,
                    }
                )

                # Add nodes per rank
                for label in (src_label, dst_label):
                    rank_key = label.split("_")[0]
                    taxon = label.split("_", 1)[1]
                    if taxon not in nodes_by_rank[rank_key]:
                        nodes_by_rank[rank_key].append(taxon)
                        nodes_count[rank_key].append(count)
                    else:
                        idx = nodes_by_rank[rank_key].index(taxon)
                        nodes_count[rank_key][idx] += count

            # Create DataFrame of flows and save
            src_tar_df = pd.DataFrame(src_tar_records)
            src_tar_df.to_csv(os.path.join(des, f"{sample_id}_flow.csv"), index=False)

            # Construct node list per rank for Sankey
            sankey_nodes: List[List[Tuple[str, int, Dict[str, Any]]]] = []
            for r in RANKS:
                rank_nodes = []
                for taxon, cnt in zip(nodes_by_rank[r], nodes_count[r]):
                    rank_nodes.append((f"{r}_{taxon}", cnt, {"color": "grey"}))
                sankey_nodes.append(rank_nodes)

            max_nodes = max(len(nodelist) for nodelist in sankey_nodes)
            fig_height = max(max_nodes / 2 * vertical_scale, 10)
            fig, ax = plt.subplots(figsize=(30, fig_height))

            sankey_plot = Sankey(
                flows=flows,
                nodes=sankey_nodes,
                flow_color_mode="source",
                node_opts={
                    "label_format": "{label} ({value:.0f})",
                    "label_opts": {"fontsize": 16},
                    "label_pos": "right",
                },
                label_pad_x=10,
                scale=0.1,
            )
            sankey_plot.draw(ax=ax)

            # Adjust labels to remove rank prefix
            for tax in tax_list:
                nodes_found = sankey_plot.find_node(tax)
                if nodes_found:
                    node = nodes_found[0]
                    node.label = tax.split("_", 1)[1]

            ax.set_title(f"{sample_id} Taxonomic Assignment", fontsize=25)
            output_svg = os.path.join(des, f"{sample_id}_sankey.svg")
            fig.savefig(output_svg, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Deprecated Sankey saved: {output_svg}")
    def _ensure_amplicon_sorter_installed(self, platform_override: Optional[str] = None) -> str:
        """
        Ensure Amplicon_sorter script is downloaded under lib_path/bin.
        If missing, download appropriate version for the platform:
          - Linux/Unix/Mac (non-M1): amplicon_sorter.py (multiprocessing)
          - Windows/Mac M1:    amplicon_sorter_single.py (single-core)
        Returns the path to the amplicon_sorter script.
        """
        import platform

        bin_dir = os.path.join(self.lib_path, "bin")
        os.makedirs(bin_dir, exist_ok=True)

        # Determine actual platform unless override is explicitly set to "windows", "linux", or "darwin"
        if platform_override and platform_override.lower() in ["windows", "linux", "darwin"]:
            sys_plat = platform_override.lower()
        else:
            sys_plat = platform.system().lower()

        # Choose single-core version on Windows or macOS ARM (M1)
        if sys_plat.startswith("win") or (sys_plat == "darwin" and platform.machine().lower().startswith("arm")):
            script_name = "amplicon_sorter_single.py"
            download_url = (
                "https://raw.githubusercontent.com/avierstr/amplicon_sorter/refs/heads/master/"
                "amplicon_sorter_single.py"
            )
        else:
            script_name = "amplicon_sorter.py"
            download_url = (
                "https://raw.githubusercontent.com/avierstr/amplicon_sorter/refs/heads/master/"
                "amplicon_sorter.py"
            )

        script_path = os.path.join(bin_dir, script_name)
        if not os.path.isfile(script_path):
            logger.info(f"Amplicon_sorter not found for '{sys_plat}'. Downloading from {download_url}")
            try:
                resp = requests.get(download_url, timeout=60)
                resp.raise_for_status()
                with open(script_path, "w", encoding="utf-8") as f_out:
                    f_out.write(resp.text)
                os.chmod(script_path, 0o755)
                logger.info(f"Downloaded Amplicon_sorter to '{script_path}'")
            except Exception as e:
                logger.error(f"Failed to download Amplicon_sorter: {e}")
                raise

        return script_path
    def amplicon_sorter(
        self,
        src: str,
        des: str,
        platform: str = "auto",     # "auto", "windows", "linux", or "darwin"
        input_format: str = "both",
        output_format: str = "both",
        minlength: int = 300,
        maxlength: Optional[int] = None,
        maxreads: int = 10000,
        allreads: bool = True,
        random_batches: bool = False,
        alignment: bool = True,      # always enforce alignment
        ambiguous: bool = False,
        length_diff_consensus: float = 15.0,
        similar_genes: float = 80.0,
        similar_species_groups: Optional[float] = 99.0,
        similar_species: float = 85.0,
        similar_consensus: float = 99.0,
        histogram_only: bool = False,
        macos_mode: bool = False,
        suppress_output: bool = True,
        tmp: str = ""
    ) -> str:
        """
        Run Amplicon_sorter to cluster sequences and extract consensus and alignments.
        - platform: "auto", "windows", "linux", or "darwin".
        - input_format: "fasta", "fastq" or "both".
        - output_format: "fasta", "fastq", or "both".
        - minlength, maxlength: filter read length range.
        - maxreads: maximum reads to process.
        - allreads: use all reads between length limits.
        - random_batches: random batch sampling (--random).
        - alignment: always True to produce alignment files (--alignment).
        - ambiguous: produce consensus with ambiguous nucleotides (--ambiguous).
        - length_diff_consensus: % length difference to merge groups (--length_diff_consensus).
        - similar_genes: % similarity to group genes (--similar_genes).
        - similar_species_groups: % similarity to create species-level groups (--similar_species_groups).
        - similar_species: % similarity to add reads to species groups (--similar_species).
        - similar_consensus: % similarity to merge consensus (--similar_consensus).
        - histogram_only: only produce read-length histogram (--histogram_only).
        - macos_mode: macOS M1 mode (--macOS).
        - suppress_output: hide command-line output.
        - tmp: temporary working directory (defaults to self.TEMP/tmp).

        Inputs:
            src: Path to a file or directory of FASTA/FASTQ to cluster.
            des: Directory where final consensus/alignments will be saved.
        Returns:
            Absolute path to des.
        """
        import glob
        import re

        # Ensure Amplicon_sorter script is available
        script_path = self._ensure_amplicon_sorter_installed(platform_override=platform)

        # Validate input/output formats
        io_fmt = self._check_input_output(input_format=input_format, output_format=output_format)

        os.makedirs(des, exist_ok=True)
        abs_des = os.path.abspath(des)

        for entry in os.scandir(src):
            if not entry.is_file():
                continue

            sample_id, ext = os.path.splitext(entry.name)
            ext = ext.lstrip(".").lower()
            # Only process files whose extension is in allowed input list
            if ext not in io_fmt["input"]:
                logger.info(f"Skipping {entry.name}: extension .{ext} not allowed by input_format={input_format}")
                continue

            logger.info(f"Processing sample: {sample_id} ({entry.name})")
            # Prepare temporary directory
            self._clean_temp()
            tmp_dir = tmp or os.path.join(self.TEMP, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            # Copy or rename input into tmp as .fasta if needed
            if ext in self.fasta_ext and ext != "fasta":
                temp_fasta = os.path.join(tmp_dir, f"{sample_id}.fasta")
                with open(entry.path, "r") as src_fh, open(temp_fasta, "w") as dst_fh:
                    for line in src_fh:
                        dst_fh.write(line)
                input_arg = f"\"{temp_fasta}\""
                logger.info(f"Copied {entry.name} → {temp_fasta}")
            else:
                input_arg = f"\"{entry.path}\""

            # Set Amplicon_sorter output to tmp/amp_out
            amp_out = os.path.join(tmp_dir, "amp_out")
            os.makedirs(amp_out, exist_ok=True)

            # Build command-line for Amplicon_sorter
            cmd_parts = [
                f"python \"{script_path}\"",
                f"-i {input_arg}",
                f"-o \"{amp_out}\"",
                f"-min {minlength}",
                f"-maxr {maxreads}",
                "--alignment"
            ]
            if maxlength is not None:
                cmd_parts.append(f"-max {maxlength}")
            if allreads:
                cmd_parts.append("--allreads")
            if random_batches:
                cmd_parts.append("--random")
            if ambiguous:
                cmd_parts.append("--ambiguous")
            cmd_parts.append(f"--length_diff_consensus {length_diff_consensus}")
            cmd_parts.append(f"--similar_genes {similar_genes}")
            if similar_species_groups is not None:
                cmd_parts.append(f"--similar_species_groups {similar_species_groups}")
            cmd_parts.append(f"--similar_species {similar_species}")
            cmd_parts.append(f"--similar_consensus {similar_consensus}")
            if histogram_only:
                cmd_parts.append("--histogram_only")
            if macos_mode:
                cmd_parts.append("--macOS")

            full_cmd = " ".join(cmd_parts)
            logger.info(f"Running Amplicon_sorter: {full_cmd}")
            self._exec(full_cmd, suppress_output=suppress_output)

            # Alignment files reside in tmp/amp_out/<sample_id>/
            sample_tmp_out = os.path.join(amp_out, sample_id)
            aln_pattern = os.path.join(sample_tmp_out, f"{sample_id}_*_*_alignment.fasta")
            aln_files = glob.glob(aln_pattern)
            if not aln_files:
                logger.error(f"No alignment files found under {sample_tmp_out}. "
                             f"Possible error in Amplicon_sorter execution.")
                continue

            # Use abs_des as final output directory
            sample_out = abs_des
            os.makedirs(sample_out, exist_ok=True)

            for aln_fp in aln_files:
                # Extract cluster number using regex to handle underscores in sample_id
                basename = os.path.basename(aln_fp)
                match = re.match(r'^(.+)_([0-9]+)_([0-9]+)_alignment\.fasta$', basename)
                if not match:
                    logger.error(f"Unexpected alignment filename format: {basename}")
                    continue

                # match.group(1) is the sample prefix (may include underscores)
                # match.group(2) is cluster index
                # match.group(3) is subbatch index
                cluster_no = match.group(2)

                # Read all records from alignment file
                with open(aln_fp, "r") as afh:
                    records = list(self._fasta_reader(afh))

                if not records:
                    logger.error(f"Empty alignment file: {basename}")
                    continue

                # Identify and remove consensus record
                consensus_record = None
                other_records = []
                for rec in records:
                    if rec["title"] == "consensus":
                        consensus_record = rec
                    else:
                        other_records.append(rec)

                if consensus_record is None:
                    logger.error(f"No 'consensus' record in {basename}")
                    continue

                # Calculate number of reads used to build consensus
                total_seqs = len(records)
                consensus_count = total_seqs - 1

                # Write consensus to output
                con_fname = f"con_{sample_id}_cluster_{cluster_no}_r{consensus_count}.fas"
                con_path = os.path.join(sample_out, con_fname)
                with open(con_path, "w") as cf:
                    title = os.path.splitext(con_fname)[0]
                    # consensus contains gaps; remove them before writing
                    cf.write(f">{title}\n{consensus_record['seq'].replace('-', '')}\n")
                logger.info(f"Wrote consensus: {con_path}")

                # Write remaining alignment sequences to output
                aln_out_fname = f"aln_{sample_id}_cluster_{cluster_no}_r{consensus_count}.fas"
                aln_out_path = os.path.join(sample_out, aln_out_fname)
                with open(aln_out_path, "w") as af:
                    for rec in other_records:
                        af.write(f">{rec['title']}\n{rec['seq']}\n")
                logger.info(f"Wrote alignment (without consensus): {aln_out_path}")

            # Verify that output files were created
            if not any(os.scandir(sample_out)):
                logger.error(f"No output written for {sample_id} in {sample_out}. "
                             f"Check for errors in Amplicon_sorter.")
        return abs_des
