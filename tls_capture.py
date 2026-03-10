#!/usr/bin/env python3
"""
TLS Traffic Capture - LIVE + DATASET SUPPORT

- Used by FastAPI for single URL prediction (capture_single_url)
- Can still be used in CLI mode for batch CSV capture if needed
"""

import subprocess
import time
import os
import csv
import logging
import shutil
import uuid
import signal
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


# ---------------- CONFIG ----------------
PCAP_DIR = Path("./scripts/pcap_files")
XML_DIR = Path("./scripts/xml_files")
LOG_FILE = "./capture_debug.log"
PROCESSED_LOG = Path("./scripts/processed_urls.txt")
TIMEOUT = 10
CAPTURE_DURATION = 7
# CHROMEDRIVER_PATH = "/opt/homebrew/bin/chromedriver"  # adjust if needed

# ----------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


class TLSTrafficCapture:
    def __init__(self):
        self.pcap_dir = PCAP_DIR
        self.xml_dir = XML_DIR
        self.processed_log = PROCESSED_LOG
        self.pcap_dir.mkdir(parents=True, exist_ok=True)
        self.xml_dir.mkdir(parents=True, exist_ok=True)
        self.chrome_temps = []
        self.kill_all_chrome()

        # For dataset mode
        self.processed_indices = set()
        if self.processed_log.exists():
            with open(self.processed_log, "r") as f:
                self.processed_indices = set(f.read().splitlines())
            logging.info(f"Loaded {len(self.processed_indices)} already processed URLs")

    # ---------- UTILITIES ----------

    def kill_all_chrome(self):
        try:
            subprocess.run(["pkill", "-9", "chrome"], stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-9", "chromedriver"], stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-9", "tcpdump"], stderr=subprocess.DEVNULL)
            time.sleep(2)
        except Exception:
            pass

    def create_chrome_temp_dir(self):
        temp_locations = [
            Path.home() / "fyp" / "chrome_temp" / f"{uuid.uuid4().hex[:8]}",
            Path("./chrome_temp") / f"{uuid.uuid4().hex[:8]}",
            Path("/tmp") / f"chrome_{uuid.uuid4().hex[:8]}",
            Path("/dev/shm") / f"chrome_{uuid.uuid4().hex[:8]}",
            ]
        for temp_path in temp_locations:
            try:
                temp_path.mkdir(parents=True, exist_ok=True)
                test_file = temp_path / "test"
                test_file.touch()
                test_file.unlink()

                self.chrome_temps.append(temp_path)
                logging.info(f"Using temp dir: {temp_path}")
                return str(temp_path)
            except Exception as e:
                logging.debug(f"Failed to create {temp_path}: {e}")
                continue

        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="chrome_")
        self.chrome_temps.append(Path(temp_dir))
        logging.info(f"Using system temp: {temp_dir}")
        return temp_dir

    def cleanup_chrome_temps(self):
        for temp_dir in self.chrome_temps:
            try:
                temp_path = Path(temp_dir) if isinstance(temp_dir, str) else temp_dir
                if temp_path.exists():
                    shutil.rmtree(temp_path, ignore_errors=True)
            except Exception:
                pass
        self.chrome_temps = []

    def check_tcpdump_permissions(self):
        try:
            result = subprocess.run(
                ["sudo", "-n", "tcpdump", "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logging.info("✓ Sudo tcpdump access confirmed")
                return True
            else:
                logging.error("✗ Cannot run tcpdump with sudo")
                logging.info("Run: sudo visudo")
                logging.info("Add: yourusername ALL=(ALL) NOPASSWD: /usr/bin/tcpdump")
                return False
        except Exception:
            logging.error("✗ tcpdump not found or sudo not configured")
            return False

    def get_active_interface(self):
        try:
            result = subprocess.run(
                ["ip", "route", "get", "8.8.8.8"],
                capture_output=True,
                text=True,
            )
            parts = result.stdout.split()
            if "dev" in parts:
                idx = parts.index("dev")
                interface = parts[idx + 1]
                logging.info(f"Active interface: {interface}")
                return interface
        except Exception:
            pass

        logging.info("Using 'any' interface")
        return "any"

    def setup_selenium(self, chrome_temp_dir):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        chrome_options.add_argument("--window-size=1280,720")
        chrome_options.add_argument("--remote-debugging-port=0")
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument(f"--user-data-dir={chrome_temp_dir}")
        chrome_options.add_argument(f"--disk-cache-dir={chrome_temp_dir}/cache")

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(TIMEOUT)
            logging.info("✓ Chrome initialized")
            return driver
        except Exception as e:
            logging.error(f"Failed to initialize Chrome: {e}")
            return None

    def start_tcpdump(self, pcap_file, interface="any"):
        try:
            cmd = [
                "sudo",
                "tcpdump",
                "-i",
                interface,
                "-w",
                str(pcap_file),
                "-U",
                "-s",
                "0",
                "-q",
            ]
            logging.debug(f"Starting tcpdump: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )

            time.sleep(3)

            if process.poll() is not None:
                stderr = process.stderr.read().decode()
                logging.error(f"tcpdump failed: {stderr}")
                return None

            logging.debug("tcpdump started successfully")
            return process
        except Exception as e:
            logging.error(f"Failed to start tcpdump: {e}")
            return None

    def stop_tcpdump(self, process):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
        except Exception as e:
            logging.warning(f"Error stopping tcpdump: {e}")
            try:
                process.kill()
            except Exception:
                pass

    def analyze_pcap(self, pcap_file):
        try:
            result = subprocess.run(
                ["tcpdump", "-r", str(pcap_file), "-n", "-c", "5"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout:
                lines = [l for l in result.stdout.split("\n") if l.strip()]
                packet_count = len(lines)
                logging.info(f"PCAP contains ~{packet_count}+ packets")
                return packet_count
            return 0
        except Exception:
            return 0

    def convert_pcap_to_xml(self, pcap_file, xml_file):
        try:
            cmd = [
                "tshark",
                "-r",
                str(pcap_file),
                "-T",
                "pdml",
                "-Y",
                "tls or ssl or tcp.port==443 or dns",
            ]

            logging.info("Converting PCAP to XML...")

            with open(xml_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )

            if result.returncode == 0:
                file_size = xml_file.stat().st_size
                if file_size > 1000:
                    logging.info(f"✓ XML created: {file_size:,} bytes")
                    return True
                else:
                    logging.warning(f"XML too small: {file_size} bytes")
                    xml_file.unlink()
                    return False
            else:
                error = result.stderr.decode()
                logging.error(f"tshark failed: {error[:200]}")
                return False
        except subprocess.TimeoutExpired:
            logging.error("tshark timeout (file too large?)")
            if xml_file.exists():
                xml_file.unlink()
            return False
        except Exception as e:
            logging.error(f"Error converting to XML: {e}")
            return False

    # ---------- LIVE PREDICTION METHOD ----------

    def capture_single_url(self, url: str):
        """
        Capture TLS traffic for a single URL and return XML path,
        or None on failure.
        """
        index = f"live_{int(time.time())}"
        base_filename = f"live_{index}"
        pcap_file = self.pcap_dir / f"{base_filename}.pcap"
        xml_file = self.xml_dir / f"{base_filename}.xml"

        logging.info("\n" + "=" * 70)
        logging.info(f"[LIVE] {url}")
        logging.info("=" * 70)

        chrome_temp = self.create_chrome_temp_dir()
        interface = self.get_active_interface()
        tcpdump_process = self.start_tcpdump(pcap_file, interface)
        if not tcpdump_process:
            logging.error("Failed to start tcpdump")
            return None

        driver = None
        try:
            driver = self.setup_selenium(chrome_temp)
            if not driver:
                logging.error("Failed to initialize Chrome")
                self.stop_tcpdump(tcpdump_process)
                return None

            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            logging.info("Loading page...")
            driver.get(url)
            time.sleep(CAPTURE_DURATION)

            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            except Exception:
                pass

            logging.info("✓ Page load attempt complete")

        except TimeoutException:
            logging.warning("Timeout loading page")
        except WebDriverException as e:
            logging.warning(f"WebDriver error: {str(e)[:150]}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
            logging.info("Stopping packet capture...")
            self.stop_tcpdump(tcpdump_process)
            time.sleep(2)
            self.cleanup_chrome_temps()

        if not pcap_file.exists():
            logging.error("✗ No PCAP file created")
            return None

        file_size = pcap_file.stat().st_size
        logging.info(f"PCAP size: {file_size:,} bytes")

        if file_size < 500:
            logging.error("✗ PCAP too small (no traffic captured)")
            pcap_file.unlink(missing_ok=True)
            return None

        packet_count = self.analyze_pcap(pcap_file)
        if packet_count == 0:
            logging.error("✗ No packets in PCAP")
            pcap_file.unlink(missing_ok=True)
            return None

        if self.convert_pcap_to_xml(pcap_file, xml_file):
            pcap_file.unlink(missing_ok=True)
            logging.info(f"✓ LIVE SUCCESS: {xml_file}")
            return str(xml_file)
        else:
            logging.error("✗ XML conversion failed")
            pcap_file.unlink(missing_ok=True)
            if xml_file.exists():
                xml_file.unlink()
            return None


# Optional: CLI for dataset mode (you already had this)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TLS Traffic Capture")
    parser.add_argument("csv_file", nargs="?", help="CSV file with url,label columns")
    parser.add_argument("--max-urls", type=int, help="Max URLs to process")
    args = parser.parse_args()

    capturer = TLSTrafficCapture()
    if not capturer.check_tcpdump_permissions():
        print("ERROR: tcpdump sudo access required")
        exit(1)

    if args.csv_file:
        # keep your existing CSV processing logic here if you want
        print("Dataset mode not fully implemented in this trimmed example.")
    else:
        print("Run via FastAPI / capture_single_url for live mode.")
