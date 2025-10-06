# Using this command to run this .py file
# python ./clean_date.py ./predicted_dates.csv -o ./cleaned_dates.csv

import pandas as pd
import re
from datetime import datetime
import argparse

# df = pd.read_csv('predicted_dates.csv')

# regular expression for dates
date_patterns = [
    r'\b(\d{1,2})(?:\s*er)?\s*(janvier|Janvier|février|Février|mars|avril|mai|juin|Juin|juillet|août|septembre|octobre|Octobre|novembre|Novembre|décembre|JANVIER|FÉVRIER|FEVRIER|MARS|AVRIL|MAI|JUIN|JUILLET|AOÛT|SEPTEMBRE|OCTOBRE|NOVEMBRE|DÉCEMBRE|DECEMBRE)\s*(\d{4})?\b',  # “1er juillet 2023” or “10 FÉVRIER”
    r'\b(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})\b',  # “02/02/2023” or “2/2/20”
    r'\b(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{2}|\d{4})\b',  # “02-02-2023” or “27-02-20”
    r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b',  # “2023-10-01” or “23-10-01”
    r'\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|Octobre|novembre|Novembre|décembre|JANVIER|FÉVRIER|MARS|AVRIL|MAI|JUIN|JUILLET|AOÛT|SEPTEMBRE|OCTOBRE|NOVEMBRE|DÉCEMBRE)\s+(\d{4})\b'  # “OCTOBRE 2022” or “octobre 2023”
]

# convert month to number format
month_map = {
    "janvier": "01", "Janvier": "01", "JANVIER": "01",
    "février": "02", "Février": "02", "FÉVRIER": "02", "FEVRIER": "02", 
    "mars": "03", "Mars": "03", "MARS": "03",
    "avril": "04", "Avril": "04", "AVRIL": "04",
    "mai": "05", "Mai": "05", "MAI": "05",
    "juin": "06", "Juin": "06", "JUIN": "06",
    "juillet": "07", "Juillet": "07", "JUILLET": "07",
    "août": "08", "Août": "08", "AOÛT": "08",
    "septembre": "09", "Septembre": "09", "SEPTEMBRE": "09",
    "octobre": "10", "Octobre": "10", "OCTOBRE": "10",
    "novembre": "11", "Novembre": "11", "NOVEMBRE": "11", 
    "décembre": "12", "Décembre": "12", "DÉCEMBRE": "12", "DECEMBRE": "12"   
}

# extract date from predicted_dates
def extract_date(text):
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

# clean the date into a unified format from extracted_dates
def clean_date(date_str):
    if not date_str:
        return None
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                day, month, year = groups
            elif len(groups) == 2:
                day = "01"
                month, year = groups
            else:
                continue      
            month = month_map.get(month, month)    
            if year and len(year) == 2:
                year = "20" + year  # assume the year is after 2000
            try:
                return f"{int(day):02}/{month}/{year}"  # "DD/MM/YYYY"
            except ValueError:
                return None
    return None

#### main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str, help="input csv file")
    parser.add_argument("-o", "--output_csv", type=str, help="output csv file",required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df['extracted_date'] = df['predicted_time'].apply(lambda x: extract_date(str(x)))
    df['cleaned_prediction_date'] = df['extracted_date'].apply(lambda x: clean_date(x))
    df['cleaned_gold_label'] = df['Gold_label'].apply(lambda x: clean_date(x))
    # print(df.columns.to_list())
    # print(df[['predicted_time', 'extracted_date', 'cleaned_date']])
    df = df[['doc_id', 'url', 'cache', 'text version', 'nature', 'published', 'entity', 'entity_type', 'Gold_label','cleaned_prediction_date','cleaned_gold_label']]
    df.to_csv(args.output_csv, index=False)
