"""
Data Loader Module
Downloads and validates the UCI Online Retail dataset
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles downloading and loading the UCI Online Retail dataset
    """
    
    UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.data_dir / "online_retail.xlsx"
    
    def download_dataset(self, force: bool = False) -> Path:
        """
        Download UCI Online Retail dataset
        
        Args:
            force: Force re-download even if file exists
            
        Returns:
            Path to downloaded file
        """
        if self.data_path.exists() and not force:
            logger.info(f"Dataset already exists at {self.data_path}")
            return self.data_path
        
        logger.info(f"Downloading dataset from {self.UCI_URL}")
        
        try:
            response = requests.get(self.UCI_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.data_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Dataset downloaded successfully to {self.data_path}")
            return self.data_path
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def load_dataset(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load dataset from local file
        
        Args:
            nrows: Number of rows to load (None = all)
            
        Returns:
            DataFrame with retail transactions
        """
        if not self.data_path.exists():
            logger.warning("Dataset not found locally. Downloading...")
            self.download_dataset()
        
        logger.info(f"Loading dataset from {self.data_path}")
        
        try:
            df = pd.read_excel(
                self.data_path,
                nrows=nrows,
                engine='openpyxl'
            )
            
            logger.info(f"Loaded {len(df):,} transactions")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, dict]:
        """
        Validate dataset structure and quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        required_columns = [
            'InvoiceNo', 'StockCode', 'Description', 
            'Quantity', 'InvoiceDate', 'UnitPrice', 
            'CustomerID', 'Country'
        ]
        
        report = {
            'total_rows': len(df),
            'columns_present': all(col in df.columns for col in required_columns),
            'missing_columns': [col for col in required_columns if col not in df.columns],
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_invoices': df['InvoiceNo'].nunique() if 'InvoiceNo' in df.columns else 0,
            'unique_products': df['StockCode'].nunique() if 'StockCode' in df.columns else 0,
            'unique_customers': df['CustomerID'].nunique() if 'CustomerID' in df.columns else 0,
            'date_range': (
                df['InvoiceDate'].min(), 
                df['InvoiceDate'].max()
            ) if 'InvoiceDate' in df.columns else (None, None)
        }
        
        # Validation checks
        is_valid = (
            report['columns_present'] and
            report['unique_invoices'] > 1000 and
            report['unique_products'] > 100 and
            report['null_percentages'].get('CustomerID', 100) < 30
        )
        
        # Log validation results
        logger.info("=" * 60)
        logger.info("DATASET VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Rows: {report['total_rows']:,}")
        logger.info(f"Unique Invoices: {report['unique_invoices']:,}")
        logger.info(f"Unique Products: {report['unique_products']:,}")
        logger.info(f"Unique Customers: {report['unique_customers']:,}")
        logger.info(f"Date Range: {report['date_range'][0]} to {report['date_range'][1]}")
        logger.info(f"Duplicate Rows: {report['duplicate_rows']:,}")
        logger.info("\nMissing Data:")
        for col, pct in report['null_percentages'].items():
            if pct > 0:
                logger.info(f"  {col}: {pct:.2f}%")
        logger.info("=" * 60)
        logger.info(f"Validation Status: {'PASSED ✓' if is_valid else 'FAILED ✗'}")
        logger.info("=" * 60)
        
        return is_valid, report
    
    def get_dataset_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate comprehensive dataset summary statistics
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'date_range': {
                'start': df['InvoiceDate'].min(),
                'end': df['InvoiceDate'].max(),
                'days': (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days
            },
            'transactions': {
                'total': df['InvoiceNo'].nunique(),
                'avg_items_per_transaction': df.groupby('InvoiceNo').size().mean(),
                'max_items_per_transaction': df.groupby('InvoiceNo').size().max()
            },
            'products': {
                'total_unique': df['StockCode'].nunique(),
                'most_common': df['StockCode'].value_counts().head(10).to_dict()
            },
            'customers': {
                'total_unique': df['CustomerID'].dropna().nunique(),
                'avg_transactions_per_customer': df.groupby('CustomerID').size().mean(),
                'countries': df['Country'].value_counts().head(10).to_dict()
            },
            'financial': {
                'total_quantity': df['Quantity'].sum(),
                'total_revenue': (df['Quantity'] * df['UnitPrice']).sum(),
                'avg_unit_price': df['UnitPrice'].mean(),
                'avg_transaction_value': df.groupby('InvoiceNo').apply(
                    lambda x: (x['Quantity'] * x['UnitPrice']).sum()
                ).mean()
            }
        }
        
        return summary


def main():
    """
    Main execution for standalone testing
    """
    # Initialize loader
    loader = DataLoader()
    
    # Download dataset
    print("Step 1: Downloading dataset...")
    loader.download_dataset()
    
    # Load dataset
    print("\nStep 2: Loading dataset...")
    df = loader.load_dataset()
    
    # Validate dataset
    print("\nStep 3: Validating dataset...")
    is_valid, report = loader.validate_dataset(df)
    
    # Get summary
    print("\nStep 4: Generating summary...")
    summary = loader.get_dataset_summary(df)
    
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Shape: {summary['shape']}")
    print(f"Memory: {summary['memory_usage_mb']:.2f} MB")
    print(f"\nDate Range: {summary['date_range']['days']} days")
    print(f"Transactions: {summary['transactions']['total']:,}")
    print(f"Products: {summary['products']['total_unique']:,}")
    print(f"Customers: {summary['customers']['total_unique']:,}")
    print(f"\nTotal Revenue: ${summary['financial']['total_revenue']:,.2f}")
    print(f"Avg Transaction Value: ${summary['financial']['avg_transaction_value']:.2f}")
    print(f"{'='*60}")
    
    if is_valid:
        print("\n✓ Dataset is ready for analysis!")
    else:
        print("\n✗ Dataset validation failed. Check report for details.")
    
    return df


if __name__ == "__main__":
    df = main()
