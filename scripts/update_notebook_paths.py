"""
Utility to update path references in notebooks to make them working-directory independent,
while strictly preserving execution order, outputs, and metadata.
"""

import json
from pathlib import Path

def update_notebooks():
    # 1. Update notebooks/phase1_cleaning_notebook.ipynb
    nb1_path = Path('notebooks/phase1_cleaning_notebook.ipynb')
    if nb1_path.exists():
        with open(nb1_path, 'r', encoding='utf-8') as f:
            nb1 = json.load(f)
        for c in nb1.get('cells', []):
            if c.get('cell_type') == 'code':
                src = ''.join(c.get('source', []))
                if 'excel_path =' in src:
                    c['source'] = [
                        "import os\n",
                        "from pathlib import Path\n",
                        "candidates = [\n",
                        "    Path('data/raw/FreshBasket_Loyalty_Churn_Dataset.xlsx'),\n",
                        "    Path('../data/raw/FreshBasket_Loyalty_Churn_Dataset.xlsx'),\n",
                        "    Path('FreshBasket_Loyalty_Churn_Dataset.xlsx'),\n",
                        "    Path('../FreshBasket_Loyalty_Churn_Dataset.xlsx')\n",
                        "]\n",
                        "excel_path = str(next((c for c in candidates if c.exists()), candidates[0]))\n",
                        "\n",
                        "# Load original raw tables\n",
                        "cust_raw = pd.read_excel(excel_path, sheet_name='fb Customer Profile', header=1)\n",
                        "churn_raw = pd.read_excel(excel_path, sheet_name='fb Churn Label', header=1)\n",
                        "act_raw = pd.read_excel(excel_path, sheet_name='fb Monthly Activity', header=1)\n",
                        "\n",
                        "print('Raw Shapes:')\n",
                        "print('  fb Customer Profile:', cust_raw.shape)\n",
                        "print('  fb Churn Label:     ', churn_raw.shape)\n",
                        "print('  fb Monthly Activity:', act_raw.shape)\n"
                    ]
                elif 'cust_clean.to_csv' in src:
                    c['source'] = [
                        "from pathlib import Path\n",
                        "out_dirs = [Path('outputs/eda/cleaned_data'), Path('../outputs/eda/cleaned_data'), Path('data/interim'), Path('../data/interim')]\n",
                        "out_dir = next((d for d in out_dirs if d.parent.exists()), Path('outputs/eda/cleaned_data'))\n",
                        "out_dir.mkdir(parents=True, exist_ok=True)\n",
                        "\n",
                        "cust_clean.to_csv(out_dir / 'cleaned_fb_customer_profile.csv', index=False)\n",
                        "churn_clean.to_csv(out_dir / 'cleaned_fb_churn_label.csv', index=False)\n",
                        "act_clean.to_csv(out_dir / 'cleaned_fb_monthly_activity.csv', index=False)\n",
                        "print(f'Cleaned CSV files successfully written to {out_dir}.')\n"
                    ]
        with open(nb1_path, 'w', encoding='utf-8') as f:
            json.dump(nb1, f, indent=1)
        print("Updated notebooks/phase1_cleaning_notebook.ipynb")

    # 2. Update notebooks/phase2_feature_engineering_notebook.ipynb
    nb2_path = Path('notebooks/phase2_feature_engineering_notebook.ipynb')
    if nb2_path.exists():
        with open(nb2_path, 'r', encoding='utf-8') as f:
            nb2 = json.load(f)
        for c in nb2.get('cells', []):
            if c.get('cell_type') == 'code':
                src = ''.join(c.get('source', []))
                if 'cleaned_fb_customer_profile.csv' in src and 'pd.read_csv' in src:
                    c['source'] = [
                        "import os\n",
                        "from pathlib import Path\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "\n",
                        "def _resolve(rel):\n",
                        "    candidates = [\n",
                        "        Path(rel),\n",
                        "        Path('../') / rel,\n",
                        "        Path('outputs/eda/cleaned_data') / Path(rel).name,\n",
                        "        Path('../outputs/eda/cleaned_data') / Path(rel).name,\n",
                        "        Path('data/interim') / Path(rel).name,\n",
                        "        Path('../data/interim') / Path(rel).name,\n",
                        "        Path('outputs/phase_1/cleaned_data') / Path(rel).name,\n",
                        "        Path('../outputs/phase_1/cleaned_data') / Path(rel).name\n",
                        "    ]\n",
                        "    return str(next((c for c in candidates if c.exists()), candidates[0]))\n",
                        "\n",
                        "cust = pd.read_csv(_resolve('data/interim/cleaned_fb_customer_profile.csv'))\n",
                        "churn = pd.read_csv(_resolve('data/interim/cleaned_fb_churn_label.csv'))\n",
                        "act = pd.read_csv(_resolve('data/interim/cleaned_fb_monthly_activity.csv'))\n",
                        "\n",
                        "act['MONTH'] = pd.to_datetime(act['MONTH'])\n",
                        "cust['SIGNUP_DATE'] = pd.to_datetime(cust['SIGNUP_DATE'])\n",
                        "cutoff_date = pd.to_datetime('2024-03-31')\n",
                        "\n",
                        "print('Loaded Cleaned Data:')\n",
                        "print('  Customer Profile: ', cust.shape)\n",
                        "print('  Churn Label:      ', churn.shape)\n",
                        "print('  Monthly Activity: ', act.shape)\n",
                        "print(f'Prediction Cutoff Date: {cutoff_date.date()}')\n"
                    ]
                elif 'Export dataset' in src:
                    c['source'] = [
                        "# Integrity assertions\n",
                        "assert len(df) == 2600, 'Error: Master row count must be exactly 2,600.'\n",
                        "assert df['CUSTOMER_ID'].nunique() == 2600, 'Error: Duplicate CUSTOMER_IDs detected.'\n",
                        "assert df['CHURNED'].value_counts()[1] == 818, 'Error: Churned count mismatch.'\n",
                        "assert df['CHURNED'].value_counts()[0] == 1782, 'Error: Retained count mismatch.'\n",
                        "\n",
                        "# Export dataset to processed data destination and legacy location\n",
                        "from pathlib import Path\n",
                        "for out_p in [Path('data/processed/phase2_modeling_dataset.csv'), Path('../data/processed/phase2_modeling_dataset.csv'), Path('phase2_modeling_dataset.csv')]:\n",
                        "    if out_p.parent.exists():\n",
                        "        df.to_csv(out_p, index=False)\n",
                        "        print(f'Exported {out_p} successfully.')\n",
                        "        break\n"
                    ]
        with open(nb2_path, 'w', encoding='utf-8') as f:
            json.dump(nb2, f, indent=1)
        print("Updated notebooks/phase2_feature_engineering_notebook.ipynb")

    # 3. Update notebooks/phase3_model_development_notebook.ipynb
    nb3_path = Path('notebooks/phase3_model_development_notebook.ipynb')
    if nb3_path.exists():
        with open(nb3_path, 'r', encoding='utf-8') as f:
            nb3 = json.load(f)
        for c in nb3.get('cells', []):
            if c.get('cell_type') == 'code':
                src = ''.join(c.get('source', []))
                if 'phase2_modeling_dataset.csv' in src and 'pd.read_csv' in src:
                    lines = c['source']
                    new_lines = []
                    for l in lines:
                        if 'df = pd.read_csv(' in l:
                            new_lines.extend([
                                "from pathlib import Path\n",
                                "candidates = [\n",
                                "    Path('data/processed/phase2_modeling_dataset.csv'),\n",
                                "    Path('../data/processed/phase2_modeling_dataset.csv'),\n",
                                "    Path('phase2_modeling_dataset.csv'),\n",
                                "    Path('../phase2_modeling_dataset.csv')\n",
                                "]\n",
                                "data_path = str(next((c for c in candidates if c.exists()), candidates[0]))\n",
                                "df = pd.read_csv(data_path)\n"
                            ])
                        else:
                            new_lines.append(l)
                    c['source'] = new_lines

                if 'val_export.to_csv' in src:
                    lines = c['source']
                    new_lines = []
                    for l in lines:
                        if 'val_export.to_csv(' in l:
                            new_lines.extend([
                                "from pathlib import Path\n",
                                "for out_p in [Path('outputs/evaluation/phase3_validation_predictions.csv'), Path('../outputs/evaluation/phase3_validation_predictions.csv'), Path('phase3_validation_predictions.csv')]:\n",
                                "    if out_p.parent.exists():\n",
                                "        val_export.to_csv(out_p, index=False)\n",
                                "        print(f\"Exported '{out_p}' successfully.\")\n",
                                "        break\n"
                            ])
                        elif 'print("Exported' in l:
                            continue  # printed in loop above
                        else:
                            new_lines.append(l)
                    c['source'] = new_lines
        with open(nb3_path, 'w', encoding='utf-8') as f:
            json.dump(nb3, f, indent=1)
        print("Updated notebooks/phase3_model_development_notebook.ipynb")

if __name__ == '__main__':
    update_notebooks()
