import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, type=str, help="Input file")
    parser.add_argument("-o", "--output_file", required=True, type=str, help="Output file")
    args = parser.parse_args()

    # Read input CSV file
    df = pd.read_csv(args.input_file)

    # Calculate accuracy between 'published' and 'cleaned_gold_label'
    df['Given_acc'] = (df['published'] == df['cleaned_gold_label']).astype(int)
    df['given_acc'] = None
    df.loc[0, 'given_acc'] = df['Given_acc'].mean() * 100

    # Calculate accuracy between 'cleaned_prediction_date' and 'cleaned_gold_label'
    df['Our_prediction_acc'] = (df['cleaned_prediction_date'] == df['cleaned_gold_label']).astype(int)
    df['our_prediction_acc'] = None
    df.loc[0, 'our_prediction_acc'] = df['Our_prediction_acc'].mean() * 100

    # Select required columns - corrected the column selection syntax
    columns_to_keep = [
        'doc_id', 'url', 'cache', 'text version', 'nature', 'entity',
        'entity_type', 'published', 'cleaned_prediction_date', 'cleaned_gold_label',
        'given_acc', 'our_prediction_acc'
    ]

    # Filter columns and handle potential missing columns
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df_filtered = df[existing_columns]

    # Rename columns
    df_filtered = df_filtered.rename(columns={
        'cleaned_prediction_date': "predicted_date",
        'cleaned_gold_label': 'gold_label'
    })

    # Print accuracies
    print(f"Accuracy from Datapolitics: {df_filtered['given_acc'].iloc[0]:.2f}%")
    print(f"Accuracy from Our prediction: {df_filtered['our_prediction_acc'].iloc[0]:.2f}%")

    # # Print available columns
    # print("\nAvailable columns:")
    # print(df_filtered.columns.tolist())

    # Save to output file
    if args.output_file.endswith('.csv'):
        output_path = args.output_file
    else:
        output_path = args.output_file + '/evaluation.csv'

    df_filtered.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
