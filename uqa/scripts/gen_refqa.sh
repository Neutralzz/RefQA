export REFQA_DATA_DIR=/root/data/refqa
 
cd ..
 
python3 wikiref_process.py --input_file wikiref.json --output_file cloze_clause_wikiref_data.json
python3 cloze2natural.py --input_file cloze_clause_wikiref_data.json --output_file refqa.json
