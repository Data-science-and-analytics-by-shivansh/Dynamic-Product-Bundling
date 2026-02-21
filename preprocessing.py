"""
Data Preprocessing Module
Cleans and transforms retail transaction data for market basket analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data cleaning and transformation for MBA
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.validation_config = self.config['validation']
    
    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw transaction data
        
        Steps:
        1. Remove cancelled/returned orders
        2. Filter invalid quantities and prices
        3. Handle missing values
        4. Remove duplicates
        5. Filter outliers
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        initial_rows = len(df)
        
        # Make a copy
        df_clean = df.copy()
        
        # 1. Remove cancelled orders (InvoiceNo starting with 'C')
        if self.data_config['remove_returns']:
            df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
            logger.info(f"Removed {initial_rows - len(df_clean):,} cancelled orders")
        
        # 2. Filter by quantity
        min_qty = self.data_config['min_quantity']
        max_qty = self.data_config['max_quantity']
        df_clean = df_clean[
            (df_clean['Quantity'] >= min_qty) & 
            (df_clean['Quantity'] <= max_qty)
        ]
        logger.info(f"Filtered by quantity ({min_qty}-{max_qty}): {len(df_clean):,} rows remaining")
        
        # 3. Filter by unit price (must be positive)
        df_clean = df_clean[df_clean['UnitPrice'] > 0]
        logger.info(f"Removed non-positive prices: {len(df_clean):,} rows remaining")
        
        # 4. Handle missing CustomerID
        null_customers = df_clean['CustomerID'].isnull().sum()
        if null_customers > 0:
            if self.data_config['handle_null_customers'] == 'drop':
                df_clean = df_clean.dropna(subset=['CustomerID'])
                logger.info(f"Dropped {null_customers:,} rows with null CustomerID")
            else:
                df_clean['CustomerID'] = df_clean['CustomerID'].fillna('GUEST')
                logger.info(f"Assigned 'GUEST' to {null_customers:,} null CustomerIDs")
        
        # 5. Remove missing descriptions
        df_clean = df_clean.dropna(subset=['Description'])
        logger.info(f"Removed null descriptions: {len(df_clean):,} rows remaining")
        
        # 6. Handle duplicates
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            if self.validation_config['handle_duplicates'] == 'keep_first':
                df_clean = df_clean.drop_duplicates(keep='first')
            elif self.validation_config['handle_duplicates'] == 'remove_all':
                df_clean = df_clean[~df_clean.duplicated(keep=False)]
            logger.info(f"Handled {duplicates:,} duplicate rows")
        
        # 7. Calculate transaction value and filter
        df_clean['TransactionValue'] = df_clean['Quantity'] * df_clean['UnitPrice']
        min_value = self.data_config['min_transaction_value']
        
        # Group by invoice and filter
        invoice_values = df_clean.groupby('InvoiceNo')['TransactionValue'].sum()
        valid_invoices = invoice_values[invoice_values >= min_value].index
        df_clean = df_clean[df_clean['InvoiceNo'].isin(valid_invoices)]
        
        logger.info(f"Filtered by min transaction value (${min_value}): {len(df_clean):,} rows remaining")
        
        # 8. Clean product descriptions (remove special characters, lowercase)
        df_clean['Description'] = df_clean['Description'].str.strip().str.upper()
        
        logger.info(f"Cleaning complete: {initial_rows:,} → {len(df_clean):,} rows ({len(df_clean)/initial_rows*100:.1f}% retained)")
        
        return df_clean
    
    def create_basket_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transaction data into basket format for MBA
        
        Format:
        - Rows: Transactions (InvoiceNo)
        - Columns: Products (StockCode)
        - Values: 1 if product in transaction, 0 otherwise
        
        Args:
            df: Cleaned transaction DataFrame
            
        Returns:
            Basket-encoded DataFrame
        """
        logger.info("Creating basket matrix...")
        
        # Group by invoice and stock code, aggregate quantities
        basket = df.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().reset_index()
        
        # Pivot to create basket matrix
        basket_matrix = basket.pivot_table(
            index='InvoiceNo',
            columns='StockCode',
            values='Quantity',
            fill_value=0
        )
        
        # Convert to binary (1 if bought, 0 if not)
        basket_encoded = (basket_matrix > 0).astype(int)
        
        logger.info(f"Basket matrix shape: {basket_encoded.shape}")
        logger.info(f"Sparsity: {(basket_encoded == 0).sum().sum() / basket_encoded.size * 100:.1f}%")
        
        return basket_encoded
    
    def create_product_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product lookup table
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Product mapping DataFrame
        """
        product_map = df[['StockCode', 'Description']].drop_duplicates()
        product_map = product_map.groupby('StockCode')['Description'].first().reset_index()
        
        # Add product statistics
        product_stats = df.groupby('StockCode').agg({
            'Quantity': 'sum',
            'UnitPrice': 'mean',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).reset_index()
        
        product_stats.columns = [
            'StockCode', 'TotalQuantitySold', 'AvgPrice', 
            'NumTransactions', 'NumCustomers'
        ]
        
        product_map = product_map.merge(product_stats, on='StockCode')
        
        logger.info(f"Created product mapping for {len(product_map):,} products")
        
        return product_map
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level features for segmentation
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Customer features DataFrame
        """
        logger.info("Creating customer features...")
        
        customer_features = df.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',
            'Quantity': 'sum',
            'TransactionValue': 'sum',
            'InvoiceDate': ['min', 'max'],
            'Country': 'first'
        }).reset_index()
        
        customer_features.columns = [
            'CustomerID', 'NumTransactions', 'TotalQuantity', 
            'TotalSpend', 'FirstPurchase', 'LastPurchase', 'Country'
        ]
        
        # Calculate derived metrics
        customer_features['AvgTransactionValue'] = (
            customer_features['TotalSpend'] / customer_features['NumTransactions']
        )
        
        customer_features['DaysSinceFirst'] = (
            customer_features['LastPurchase'] - customer_features['FirstPurchase']
        ).dt.days
        
        customer_features['PurchaseFrequency'] = (
            customer_features['NumTransactions'] / (customer_features['DaysSinceFirst'] + 1)
        )
        
        # Segment customers (simple RFM-style)
        customer_features['Segment'] = pd.cut(
            customer_features['TotalSpend'],
            bins=[0, 500, 2000, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        
        logger.info(f"Created features for {len(customer_features):,} customers")
        
        return customer_features
    
    def save_processed_data(
        self, 
        df_clean: pd.DataFrame,
        basket_encoded: pd.DataFrame,
        product_map: pd.DataFrame,
        customer_features: pd.DataFrame,
        output_dir: str = "data/processed"
    ):
        """
        Save all processed datasets
        
        Args:
            df_clean: Cleaned transactions
            basket_encoded: Basket matrix
            product_map: Product mapping
            customer_features: Customer features
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV and Parquet for flexibility
        logger.info(f"Saving processed data to {output_dir}")
        
        df_clean.to_csv(output_path / 'transactions_clean.csv', index=False)
        df_clean.to_parquet(output_path / 'transactions_clean.parquet', index=False)
        
        basket_encoded.to_csv(output_path / 'basket_encoded.csv')
        basket_encoded.to_parquet(output_path / 'basket_encoded.parquet')
        
        product_map.to_csv(output_path / 'product_mapping.csv', index=False)
        customer_features.to_csv(output_path / 'customer_features.csv', index=False)
        
        logger.info("All processed data saved successfully!")
    
    def get_preprocessing_summary(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        basket_encoded: pd.DataFrame
    ) -> dict:
        """
        Generate preprocessing summary report
        
        Args:
            df_raw: Original DataFrame
            df_clean: Cleaned DataFrame
            basket_encoded: Basket matrix
            
        Returns:
            Summary dictionary
        """
        summary = {
            'original_rows': len(df_raw),
            'cleaned_rows': len(df_clean),
            'rows_retained_pct': len(df_clean) / len(df_raw) * 100,
            'unique_transactions': df_clean['InvoiceNo'].nunique(),
            'unique_products': df_clean['StockCode'].nunique(),
            'unique_customers': df_clean['CustomerID'].nunique(),
            'basket_matrix_shape': basket_encoded.shape,
            'basket_sparsity': (basket_encoded == 0).sum().sum() / basket_encoded.size * 100,
            'avg_items_per_basket': basket_encoded.sum(axis=1).mean(),
            'date_range': (df_clean['InvoiceDate'].min(), df_clean['InvoiceDate'].max())
        }
        
        return summary


def main():
    """
    Main execution for standalone testing
    """
    from data_loader import DataLoader
    
    print("="*60)
    print("MARKET BASKET ANALYSIS - DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    print("\nStep 1: Loading raw data...")
    loader = DataLoader()
    df_raw = loader.load_dataset()
    
    # Initialize preprocessor
    print("\nStep 2: Initializing preprocessor...")
    preprocessor = DataPreprocessor()
    
    # Clean data
    print("\nStep 3: Cleaning transactions...")
    df_clean = preprocessor.clean_transactions(df_raw)
    
    # Create basket matrix
    print("\nStep 4: Creating basket matrix...")
    basket_encoded = preprocessor.create_basket_matrix(df_clean)
    
    # Create product mapping
    print("\nStep 5: Creating product mapping...")
    product_map = preprocessor.create_product_mapping(df_clean)
    
    # Create customer features
    print("\nStep 6: Creating customer features...")
    customer_features = preprocessor.create_customer_features(df_clean)
    
    # Save processed data
    print("\nStep 7: Saving processed data...")
    preprocessor.save_processed_data(
        df_clean, basket_encoded, product_map, customer_features
    )
    
    # Generate summary
    print("\nStep 8: Generating summary...")
    summary = preprocessor.get_preprocessing_summary(df_raw, df_clean, basket_encoded)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Original Rows: {summary['original_rows']:,}")
    print(f"Cleaned Rows: {summary['cleaned_rows']:,} ({summary['rows_retained_pct']:.1f}% retained)")
    print(f"Unique Transactions: {summary['unique_transactions']:,}")
    print(f"Unique Products: {summary['unique_products']:,}")
    print(f"Unique Customers: {summary['unique_customers']:,}")
    print(f"\nBasket Matrix: {summary['basket_matrix_shape']}")
    print(f"Sparsity: {summary['basket_sparsity']:.1f}%")
    print(f"Avg Items per Basket: {summary['avg_items_per_basket']:.2f}")
    print(f"{'='*60}")
    print("\n✓ Preprocessing complete! Data ready for MBA.")
    
    return df_clean, basket_encoded, product_map, customer_features


if __name__ == "__main__":
    df_clean, basket_encoded, product_map, customer_features = main()
