import os
import pandas as pd
import re

def extract_body(text):
    
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    cleaned_lines = []
    for line in body.splitlines():
        if re.match(r"^(Archive-name|From|Subject|Path|Xref|Organization|Lines|Newsgroups|Message-ID|Keywords):", line, re.I):
            continue
        if line.strip().startswith((">", "|")): 
            continue
        cleaned_lines.append(line)

    body_text = "\n".join(cleaned_lines).strip()
    body_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", body_text)
    return body_text


def convert_20ng_to_excel(root_folder, output_excel, max_files=None):
   
    data = []
    for category in sorted(os.listdir(root_folder)):
        category_path = os.path.join(root_folder, category)

        if os.path.isdir(category_path):
            print(f"Processing category: {category}")

            for i, filename in enumerate(os.listdir(category_path)):
                if max_files and i >= max_files:
                    break

                file_path = os.path.join(category_path, filename)
                try:
                    with open(file_path, "r", encoding="latin1") as f:
                        raw_text = f.read()
                        body = extract_body(raw_text)
                        if body:
                            data.append({
                                "filename": filename,
                                "category": category,
                                "text": body
                            })
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"✅ Step 1 done: Saved {len(df)} rows to {output_excel}")


def clean_body(raw_text):
    if pd.isna(raw_text):
        return ""


    parts = re.split(r"\n\s*\n", raw_text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    cleaned_lines = []
    for line in body.splitlines():
        if re.match(r"^(archive-name|from|subject|path|xref|organization|lines|newsgroups|message-id|keywords|last-modified|version):", line, re.I):
            continue
        if line.strip().startswith((">", "|")):
            continue
        if line.strip().startswith("--"):  
            break
        if re.search(r"In article\s*<.*?>", line, re.I):
            continue
        if re.search(r"writes:|wrote:", line, re.I):
            continue
        cleaned_lines.append(line)

    body = "\n".join(cleaned_lines)


    body = re.sub(r"\S+@\S+", " ", body)
    body = re.sub(r"http\S+|www\.\S+", " ", body)
    body = re.sub(r"<[^>]+>", " ", body)


    body = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", body)


    body = re.sub(r"\s{2,}", " ", body)
    body = body.lower().strip()

    return body


def process_final_dataset(input_excel, output_excel):
    df = pd.read_excel(input_excel, engine="openpyxl")
    print(f"Loaded {len(df)} rows for deep cleaning...")

    df["text"] = df["text"].apply(clean_body)

    df["text"] = df["text"].str.strip()

    df = df[df["text"].str.len() > 0]

    df = df[["category", "text"]].reset_index(drop=True)

    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"✅ Final dataset saved: {len(df)} rows → {output_excel}")

if __name__ == "__main__":
    root_folder = r"D:\Infosys Project\20news-18828"

 
    convert_20ng_to_excel(
        root_folder=root_folder,
        output_excel="20news_initial.xlsx",
        max_files=50 
    )


    process_final_dataset(
        input_excel="20news_initial.xlsx",
        output_excel="news_MetaData.xlsx"
    )

    print("\n🚀 Processing Complete!")
