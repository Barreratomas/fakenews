from sklearn.model_selection import train_test_split

def split_train_val(df, val_size=0.2, random_state=42):
    return train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],
        random_state=random_state
    )
