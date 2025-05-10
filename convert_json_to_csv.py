#!/usr/bin/env python3
"""
convert_json_to_csv.py (match-level)
~~~~~~~~~~~~~~~~~~~~~~
Convert a giant League match JSON dump into a match-level CSV.
• 1 row = 1 match
• Columns: match_id, win (1 if blue wins), blue1_*, ..., blue5_*, red1_*, ..., red5_*
"""
import argparse, ijson, csv, pathlib, itertools, json, sys

def enumerate_matches(istream):
    yield from ijson.items(istream, 'match_data.item')

def flatten_participant(p_dict, prefix):
    data = {
        f"{prefix}summonerLevel": p_dict.get("summonerLevel", 0),
        f"{prefix}championId": p_dict.get("championId", 0),
    }
    challenges = p_dict.get("challenges", {})
    for k, v in challenges.items():
        if isinstance(v, (int, float)):
            data[f"{prefix}ch_{k}"] = v
    return data

def main(inp: pathlib.Path, out: pathlib.Path):
    all_keys = set()
    match_rows = []
    with inp.open('rb') as f:
        for match in enumerate_matches(f):
            mid = match.get("gameId")
            participants = match.get("participants", [])
            if len(participants) != 10:
                continue  # skip incomplete matches
            row = {"match_id": mid}
            # Blue: idx 0-4, Red: idx 5-9
            for i in range(5):
                pdata = flatten_participant(participants[i], f"blue{i+1}_")
                row.update(pdata)
                all_keys.update(pdata.keys())
            for i in range(5, 10):
                pdata = flatten_participant(participants[i], f"red{i-4}_")
                row.update(pdata)
                all_keys.update(pdata.keys())
            # Use blue1's win as match label
            row["win"] = int(participants[0]["win"])
            match_rows.append(row)
    # Write CSV
    static_keys = ["match_id", "win"]
    player_keys = sorted(all_keys)
    fieldnames = static_keys + player_keys
    with out.open('w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in match_rows:
            writer.writerow(row)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=pathlib.Path, help="raw match dump (huge JSON)")
    ap.add_argument("-o", "--out", type=pathlib.Path, default="league_matches_matchlevel.csv")
    args = ap.parse_args()
    main(args.json_path, args.out)
