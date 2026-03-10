#!/usr/bin/env python3
"""
feature_extractor.py (backend version)

Reuses the SAME HF/NF logic you used in your original dataset
(EnhancedFeatureExtractor) to ensure training & inference match.

Output: single-row DataFrame with columns:
['HF10','HF11','HF2','HF3','HF4','HF5',
 'HF_special_chars','HF_subdomain_depth','HF_uppercase',
 'NF1','NF2','NF3','NF4','NF5','NF6',
 'NF_avg_extensions','NF_certificates','NF_cipher_diversity',
 'NF_client_hello','NF_handshake_msgs','NF_tls_version',
 'NF_ttl_std','NF_unique_handshakes']
"""

import xml.etree.ElementTree as ET
import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlparse
import logging
from collections import Counter
import statistics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# --------- HF: hostname-based features (copied from your script) ---------

def extract_hostname_features(url: str) -> dict:
    """Extract hostname-based features (HF2-HF11 + extras)"""
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
        hostname = hostname.replace('www.', '')

        # HF2: Count of digits
        hf2 = sum(c.isdigit() for c in hostname)

        # HF3: Average word length
        words = re.split(r'[.\-_]', hostname)
        words = [w for w in words if w]
        hf3 = sum(len(w) for w in words) / len(words) if words else 0

        # HF4: Longest word length
        hf4 = max(len(w) for w in words) if words else 0

        # HF5: Total length of hostname
        hf5 = len(hostname)

        # HF10: Count of dots
        hf10 = hostname.count('.')

        # HF11: Count of hyphens
        hf11 = hostname.count('-')

        # Additional hostname features
        hf_special = sum(not c.isalnum() and c not in '.-_' for c in hostname)
        hf_uppercase = sum(c.isupper() for c in hostname)
        hf_subdomain_count = hostname.count('.')  # your "subdomain depth"

        return {
            'HF2': hf2,
            'HF3': round(hf3, 2),
            'HF4': hf4,
            'HF5': hf5,
            'HF10': hf10,
            'HF11': hf11,
            'HF_special_chars': hf_special,
            'HF_uppercase': hf_uppercase,
            'HF_subdomain_depth': hf_subdomain_count,
        }
    except Exception as e:
        logging.error(f"Error extracting hostname features from {url}: {e}")
        return {k: 0 for k in [
            'HF2', 'HF3', 'HF4', 'HF5', 'HF10', 'HF11',
            'HF_special_chars', 'HF_uppercase', 'HF_subdomain_depth'
        ]}


# --------- NF: network/TLS-based features (copied from your script) ---------

def extract_network_features(xml_path: str) -> dict:
    """Extract network/TLS features from a single XML (PDML) file."""
    xml_file = Path(xml_path)
    try:
        tree = ET.parse(str(xml_file))
        root = tree.getroot()

        tls_sessions = set()
        ipv4_addresses = set()
        ipv6_addresses = set()
        ttl_values = []
        cipher_suites = []
        tls_versions = []
        server_hello_count = 0
        client_hello_count = 0
        certificate_count = 0

        handshake_types = []
        extensions_count = []

        # Parse packets
        for packet in root.findall('.//packet'):
            # IP v4
            for proto in packet.findall('.//proto[@name="ip"]'):
                src = proto.find('.//field[@name="ip.src"]')
                dst = proto.find('.//field[@name="ip.dst"]')
                if src is not None:
                    ipv4_addresses.add(src.get('show'))
                if dst is not None:
                    ipv4_addresses.add(dst.get('show'))

                ttl = proto.find('.//field[@name="ip.ttl"]')
                if ttl is not None:
                    try:
                        ttl_values.append(int(ttl.get('show')))
                    except Exception:
                        pass

            # IP v6
            for proto in packet.findall('.//proto[@name="ipv6"]'):
                src = proto.find('.//field[@name="ipv6.src"]')
                dst = proto.find('.//field[@name="ipv6.dst"]')
                if src is not None:
                    ipv6_addresses.add(src.get('show'))
                if dst is not None:
                    ipv6_addresses.add(dst.get('show'))

            # TLS
            for proto in packet.findall('.//proto[@name="tls"]'):
                # TLS Version
                version = proto.find('.//field[@name="tls.record.version"]')
                if version is not None:
                    tls_versions.append(version.get('show'))

                # Session ID
                session = proto.find('.//field[@name="tls.handshake.session_id"]')
                if session is not None:
                    sess_id = session.get('show')
                    if sess_id:
                        tls_sessions.add(sess_id)

                # Handshake type
                handshake_type = proto.find('.//field[@name="tls.handshake.type"]')
                if handshake_type is not None:
                    h_type = handshake_type.get('show')
                    handshake_types.append(h_type)

                    if h_type == '1':      # ClientHello
                        client_hello_count += 1
                    elif h_type == '2':    # ServerHello
                        server_hello_count += 1
                    elif h_type == '11':   # Certificate
                        certificate_count += 1

                # Cipher Suites
                for cipher in proto.findall('.//field[@name="tls.handshake.ciphersuite"]'):
                    cipher_val = cipher.get('show')
                    if cipher_val:
                        cipher_suites.append(cipher_val)

                # Extensions
                extensions = proto.findall('.//field[@name="tls.handshake.extension.type"]')
                if extensions:
                    extensions_count.append(len(extensions))

            # SSL (older naming)
            for proto in packet.findall('.//proto[@name="ssl"]'):
                handshake_type = proto.find('.//field[@name="ssl.handshake.type"]')
                if handshake_type is not None:
                    h_type = handshake_type.get('show')
                    if h_type == '1':
                        client_hello_count += 1
                    elif h_type == '2':
                        server_hello_count += 1

        # ---- Original NF1–NF6 from your script ----
        nf1 = max(len(tls_sessions), 1) if tls_sessions else 1
        nf2 = len(ipv4_addresses)
        nf3 = len(ipv6_addresses)
        nf4 = int(sum(ttl_values) / len(ttl_values)) if ttl_values else 64
        nf5 = len(set(cipher_suites))  # Unique cipher suites
        nf6 = server_hello_count

        # Enhanced features
        nf_client_hello = client_hello_count
        nf_certificates = certificate_count
        nf_cipher_diversity = (
            len(set(cipher_suites)) / max(len(cipher_suites), 1)
            if cipher_suites else 0
        )
        nf_avg_extensions = (
            sum(extensions_count) / len(extensions_count)
            if extensions_count else 0
        )

        # TLS version average (same logic you used)
        nf_tls_version_avg = 0
        if tls_versions:
            versions_numeric = []
            for v in tls_versions:
                try:
                    if isinstance(v, str) and '0x' in v:
                        versions_numeric.append(int(v, 16))
                except Exception:
                    pass
            if versions_numeric:
                nf_tls_version_avg = sum(versions_numeric) / len(versions_numeric)

        nf_handshake_messages = len(handshake_types)
        nf_unique_handshakes = len(set(handshake_types))

        nf_ttl_std = 0
        if len(ttl_values) > 1:
            nf_ttl_std = statistics.stdev(ttl_values)

        return {
            'NF1': nf1,
            'NF2': nf2,
            'NF3': nf3,
            'NF4': nf4,
            'NF5': nf5,
            'NF6': nf6,
            'NF_client_hello': nf_client_hello,
            'NF_certificates': nf_certificates,
            'NF_cipher_diversity': round(nf_cipher_diversity, 3),
            'NF_avg_extensions': round(nf_avg_extensions, 2),
            'NF_tls_version': round(nf_tls_version_avg, 0),
            'NF_handshake_msgs': nf_handshake_messages,
            'NF_unique_handshakes': nf_unique_handshakes,
            'NF_ttl_std': round(nf_ttl_std, 2),
        }

    except Exception as e:
        logging.error(f"Error extracting network features from {xml_file}: {e}")
        return {
            'NF1': 0, 'NF2': 0, 'NF3': 0, 'NF4': 64, 'NF5': 0, 'NF6': 0,
            'NF_client_hello': 0, 'NF_certificates': 0, 'NF_cipher_diversity': 0,
            'NF_avg_extensions': 0, 'NF_tls_version': 0, 'NF_handshake_msgs': 0,
            'NF_unique_handshakes': 0, 'NF_ttl_std': 0,
        }


# --------- MAIN ENTRY POINT FOR BACKEND ---------

def extract_features_from_url_and_xml(url: str, xml_path: str) -> pd.DataFrame:
    """
    Backend-facing API: given a URL and its captured XML (PDML),
    return ONE ROW of features with the same HF/NF logic used
    to build final_dataset_14008.csv (minus url/label).
    """
    hf = extract_hostname_features(url)
    nf = extract_network_features(xml_path)

    features = {**hf, **nf}

    expected_cols = [
        'HF10', 'HF11', 'HF2', 'HF3', 'HF4', 'HF5',
        'HF_special_chars', 'HF_subdomain_depth', 'HF_uppercase',
        'NF1', 'NF2', 'NF3', 'NF4', 'NF5', 'NF6',
        'NF_avg_extensions', 'NF_certificates', 'NF_cipher_diversity',
        'NF_client_hello', 'NF_handshake_msgs', 'NF_tls_version',
        'NF_ttl_std', 'NF_unique_handshakes',
    ]
    for col in expected_cols:
        if col not in features:
            features[col] = 0

    return pd.DataFrame([features])
    """
    Backend-facing API: given a URL and its captured XML (PDML),
    return ONE ROW of features with the same HF/NF logic used
    to build final_dataset_14008.csv (minus url/label).
    """
    hf = extract_hostname_features(url)
    nf = extract_network_features(xml_path)

    # Merge dicts
    features = {**hf, **nf}

    # Ensure all expected columns exist (in case of errors)
    expected_cols = [
        'HF10', 'HF11', 'HF2', 'HF3', 'HF4', 'HF5',
        'HF_special_chars', 'HF_subdomain_depth', 'HF_uppercase',
        'NF1', 'NF2', 'NF3', 'NF4', 'NF5', 'NF6',
        'NF_avg_extensions', 'NF_certificates', 'NF_cipher_diversity',
        'NF_client_hello', 'NF_handshake_msgs', 'NF_tls_version',
        'NF_ttl_std', 'NF_unique_handshakes',
    ]
    for col in expected_cols:
        if col not in features:
            features[col] = 0

    # Order doesn't matter here; main.py will reorder using FEATURE_COLUMNS
    return pd.DataFrame([features])
