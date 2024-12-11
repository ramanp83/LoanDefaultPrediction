    def preprocess(self, data):
        """
        Preprocess the data.
        - Handle missing values.
        - Encode categorical variables.
        - Convert text-based numbers to numeric.
        - Scale numerical features.
        Args:
        - data: DataFrame with raw data.
        """
        # Handle missing values (for simplicity, we fill with the median for numeric columns)
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

        # Convert 'loan_term' to numeric by removing the text and trimming spaces
        if 'loan_term' in data.columns:
            data['loan_term'] = data['loan_term'].replace(r'\D', '', regex=True).astype(float)

        # Encode categorical variables using LabelEncoder
        label_columns = ['home_ownership', 'purpose', 'sub_grade', 'application_type', 'verification_status']
        label_encoder = LabelEncoder()
        for col in label_columns:
            data[col] = label_encoder.fit_transform(data[col].astype(str))  # Convert to string to handle any potential non-numeric entries

        # Separate features and target variable
        X = data.drop(columns=['loan_status', 'customer_id', 'transaction_date'])
        y = data['loan_status']

        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y
