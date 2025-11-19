#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate HLTB Scraper (Threaded, Cleaned, and Resumable)
---------------------------------------------------------
Merges:
  - Robust multi-threaded, checkpointing architecture.
  - Smart pre-cleaning of SteamSpy names for a high match rate.
  - Robust 'getattr' logic to handle missing HLTB data and default to 0.
"""

import pandas as pd
import re
import time
import os
import sys
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from howlongtobeatpy import HowLongToBeat, HowLongToBeatEntry
from tqdm import tqdm  # Using tqdm for a clear progress bar

# --- Configuration ---
CHECKPOINT_FILE = "hltb_ultimate_results.csv" # Save directly to the final CSV
MAX_THREADS = 10  # HLTB can be sensitive; 10 is safer than 20.
TIME_SLEEP_MIN = 0.4  # Be polite to the server
TIME_SLEEP_MAX = 0.8

# --- 1. Define Suffix Pattern (Expanded) ---
suffixes = [
    'goty', 'game of the year', 'complete edition', 'definitive edition', 
    'remastered', 'legacy edition', 'deluxe edition', 'enhanced edition',
    'vr', 'hd', 'ultimate edition', 'gold edition'
]
suffix_pattern = r'\b(' + '|'.join(suffixes) + r')\b'

# --- 2. Define the Improved (Safer) Cleaning Function ---
def clean_name_improved(name):
    name_str = str(name).lower()
    
    # Step 1: Handle special chars and Roman numerals
    name_str = name_str.replace('&', 'and')
    name_str = name_str.replace('™', '')
    name_str = name_str.replace('®', '')
    
    name_str = name_str.replace(' ix', ' 9')
    name_str = name_str.replace(' iv', ' 4')
    name_str = name_str.replace(' v', ' 5')
    name_str = name_str.replace(' vi', ' 6')
    name_str = name_str.replace(' vii', ' 7')
    name_str = name_str.replace(' viii', ' 8')
    name_str = name_str.replace(' x', ' 10')
    name_str = name_str.replace(' i', ' 1')

    # Step 2: Remove all *remaining* punctuation
    name_str = re.sub(r'[^a-z0-9\s]', '', name_str)
    
    # Step 3: Remove common edition suffixes
    name_str = re.sub(suffix_pattern, '', name_str)
    
    # Step 4: Normalize whitespace
    name_str = re.sub(r'\s+', ' ', name_str)
    
    return name_str.strip()

# --- 3. Load and Prepare the SteamSpy Input List ---
def get_names_to_scrape(input_file="Steamspy_50k.csv"):
    print(f"Loading {input_file}...")
    try:
        base_df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"ERROR: {input_file} not found.")
        sys.exit(1)

    # Drop rows with null names
    base_df.dropna(subset=['name'], inplace=True)

    # Filter out 'noise' (DLCs, Demos, etc.) BEFORE we scrape
    noise_keywords = ['Soundtrack', 'DLC', 'Artbook', 'Demo', 'Beta', 'Prologue', 'Season Pass']
    is_noise = base_df['name'].str.contains('|'.join(noise_keywords), case=False, na=False)
    games_to_scrape_df = base_df[~is_noise].copy()

    # Apply the *improved* cleaning function
    games_to_scrape_df['clean_name'] = games_to_scrape_df['name'].apply(clean_name_improved)

    # Get a unique list of clean names to scrape
    unique_clean_names = games_to_scrape_df['clean_name'].unique()
    unique_clean_names = [name for name in unique_clean_names if name] # Remove empty strings
    
    print(f"Loaded {len(base_df)} rows, filtered down to {len(games_to_scrape_df)} valid games.")
    print(f"Found {len(unique_clean_names)} unique clean names to scrape.")
    return unique_clean_names

# --- 4. Checkpointing Functions (from your script) ---
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            ck = pd.read_csv(CHECKPOINT_FILE)
            # Use 'clean_name' as the key
            done = set(ck["clean_name"].astype(str).tolist())
            return ck.to_dict("records"), done
        except Exception as e:
            print(f"Warning: Could not read checkpoint file. Starting fresh. Error: {e}")
            return [], set()
    return [], set()

def save_checkpoint(rows, is_first_save):
    try:
        df = pd.DataFrame(rows)
        if is_first_save:
            # First save, write header
            df.to_csv(CHECKPOINT_FILE, index=False, mode='w')
        else:
            # Append, no header
            df.to_csv(CHECKPOINT_FILE, index=False, mode='a', header=False)
    except Exception as e:
        print(f"\n[!] Checkpoint save FAILED: {e}")

# --- 5. Scraping Functions (Merged Logic) ---
def fetch_times(hltb: HowLongToBeat, clean_name: str):
    """
    Fetches HLTB data for a *clean* game name.
    Returns a dictionary with 0 for missing/None values.
    """
    try:
        results = hltb.search(clean_name)
    except Exception as e:
        print(f"\n[!] Error fetching '{clean_name}': {e}")
        return None # Will be handled in the main loop

    if not results:
        return None # No match found
    
    best: HowLongToBeatEntry = results[0] # Take the best match
    
    # Use getattr(..., 0) or 0 for robust defaulting
    return {
        "clean_name": clean_name, # The key we join on
        "Name": getattr(best, "game_name", "N/A"),
        "Main Story (h)": getattr(best, "main_story", 0) or 0,
        "Main + Sides (h)": getattr(best, "main_extra", 0) or 0,
        "Completionist (h)": getattr(best, "completionist", 0) or 0,
        "All Styles (h)": getattr(best, "all_styles", 0) or 0,
        "Co-Op (h)": getattr(best, "co_op", 0) or 0, # Note: Your script had 'coop_time'
        "Vs (h)": getattr(best, "vs", 0) or 0,       # Note: Your script had 'mp_time'
    }

def threaded_fetch(hltb, name):
    """
    Wrapper for the thread pool. Includes sleep logic.
    """
    result = fetch_times(hltb, name)
    
    # Small random delay
    time.sleep(random.uniform(TIME_SLEEP_MIN, TIME_SLEEP_MAX))
    
    if result is None:
        # If fetch fails or finds no match, return a 'fail' row
        # This prevents us from re-scraping it on resume
        return {
            "clean_name": name,
            "Name": "N/A - No Match",
            "Main Story (h)": 0,
            "Main + Sides (h)": 0,
            "Completionist (h)": 0,
            "All Styles (h)": 0,
            "Co-Op (h)": 0,
            "Vs (h)": 0
        }
    return result

# --- 6. Main Execution ---
def main(input_file):
    all_names = get_names_to_scrape(input_file)
    total_names = len(all_names)
    
    all_saved_rows, done_names = load_checkpoint()
    print(f"[i] Resuming from checkpoint: {len(done_names)} names already scraped.")
    
    names_to_do = [n for n in all_names if n not in done_names]
    print(f"[i] Remaining: {len(names_to_do)} names.")
    
    if not names_to_do:
        print("All names already scraped. Exiting.")
        return

    hltb = HowLongToBeat()
    new_rows_this_session = []
    
    # This ensures we save *all* progress on exit
    def exit_handler():
        if new_rows_this_session:
            print(f"\n[!] Exiting. Saving last {len(new_rows_this_session)} new rows...")
            # Check if checkpoint file was *ever* created
            is_first = not os.path.exists(CHECKPOINT_FILE) 
            save_checkpoint(new_rows_this_session, is_first_save=is_first)
            print("[i] Save on exit complete.")
    
    import atexit
    atexit.register(exit_handler)

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Submit all jobs
        futures = {executor.submit(threaded_fetch, hltb, name): name for name in names_to_do}

        # Process as they complete, with a progress bar
        for future in tqdm(as_completed(futures), total=len(names_to_do), desc="Scraping HLTB"):
            result = future.result()
            
            if result:
                new_rows_this_session.append(result)

            # Checkpoint saving logic
            if len(new_rows_this_session) >= 50: # Save every 50 new rows
                print(f"\n[i] Checkpointing {len(new_rows_this_session)} new rows...")
                # Check if checkpoint file was *ever* created
                is_first = not os.path.exists(CHECKPOINT_FILE) 
                save_checkpoint(new_rows_this_session, is_first_save=is_first)
                # Clear the session buffer
                new_rows_this_session = []


    # Final save at the very end for any remaining rows
    if new_rows_this_session:
        print(f"\n[i] Saving final {len(new_rows_this_session)} rows...")
        is_first = not os.path.exists(CHECKPOINT_FILE)
        save_checkpoint(new_rows_this_session, is_first_save=is_first)

    total_in_file = len(done_names) + len(names_to_do) # A bit simplified, but close
    print(f"\n[✓] Done. All {len(names_to_do)} new items processed.")
    print(f"[✓] Final results are in: {CHECKPOINT_FILE}")

if __name__ == "__main__":
    # Simplified to just use the Steamspy file in the local dir
    main(input_file="Steamspy_50k.csv")