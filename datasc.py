# Run the script
# python datasc.py

# User inputs:
# Enter the path to your data file: data.csv
# Enter target column name (press Enter if none): target
# Enter API endpoint: analysis/predict
# Enter your custom prompt: Analyze this data for trends 

from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import warnings
import asyncio
import aiohttp
from datetime import datetime
import json

warnings.filterwarnings('ignore')
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DataLoader:
    """Data loader with support for common data formats"""
    
    def __init__(self):
        self.supported_formats = {'.csv', '.json', '.xlsx', '.parquet'}
        
    def validate_file(self, filepath: str) -> bool:
        """Validate if file exists and has supported format"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        if path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return True

    def load_data(self, source: Union[str, Dict], **kwargs) -> pd.DataFrame:
        """Load data with automatic format detection"""
        try:
            if isinstance(source, str):
                if self.validate_file(source):
                    path = Path(source)
                    if path.suffix == '.csv':
                        return pd.read_csv(source, **kwargs)
                    elif path.suffix == '.json':
                        return pd.read_json(source, **kwargs)
                    elif path.suffix == '.xlsx':
                        return pd.read_excel(source, **kwargs)
                    elif path.suffix == '.parquet':
                        return pd.read_parquet(source, **kwargs)
            elif isinstance(source, dict):
                return pd.DataFrame.from_dict(source)
            raise ValueError("Unsupported data source")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

class DataPreprocessor:
    """Handles data preprocessing tasks"""
    
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        df_copy = df.copy()
        df_copy = self._handle_missing_values(df_copy)
        df_copy = self._remove_duplicates(df_copy)
        df_copy = self._encode_categoricals(df_copy)
        df_copy = self._scale_numerics(df_copy)
        return df_copy
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values based on data type"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.numeric_imputer.fit_transform(df[numeric_cols])
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates()
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col] = self.label_encoder.fit_transform(df[col].astype(str))
        return df
    
    def _scale_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric variables"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

class DataAnalyzer:
    """Handles data analysis"""
    
    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """Perform basic data analysis"""
        return {
            'basic_stats': self._get_basic_stats(df),
            'correlations': self._get_correlations(df),
            'missing_values': df.isnull().sum().to_dict()
        }
    
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate basic statistics"""
        return {
            'summary': df.describe(),
            'dtypes': df.dtypes.to_dict()
        }
    
    def _get_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns"""
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        return numeric_df.corr()

class DataVisualizer:
    """Handles data visualization tasks"""
    
    def create_visualizations(self, df: pd.DataFrame) -> Dict:
        """Create various visualizations for the dataset"""
        plots = {}
        
        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            plots[f'{col}_distribution'] = fig
            
        # Correlation heatmap
        corr_matrix = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix, x=corr_matrix.columns, y=corr_matrix.columns))
        fig.update_layout(title='Correlation Heatmap')
        plots['correlation_heatmap'] = fig
        
        # Box plots for detecting outliers
        for col in numeric_cols:
            fig = px.box(df, y=col, title=f'Box Plot of {col}')
            plots[f'{col}_boxplot'] = fig
            
        return plots

class StatisticalAnalyzer:
    """Handles statistical analysis"""
    
    def perform_statistical_tests(self, df: pd.DataFrame) -> Dict:
        """Perform various statistical tests on the data"""
        results = {}
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Normality tests
        for col in numeric_cols:
            stat, p_value = stats.normaltest(df[col].dropna())
            results[f'{col}_normality'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        
        # Correlation tests
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 < col2:
                    corr, p_value = stats.pearsonr(df[col1], df[col2])
                    results[f'correlation_{col1}_{col2}'] = {
                        'correlation': corr,
                        'p_value': p_value
                    }
        
        return results

class MachineLearning:
    """Handles machine learning tasks"""
    
    def __init__(self):
        self.models = {
            'random_forest_classifier': RandomForestClassifier(),
            'random_forest_regressor': RandomForestRegressor(),
            'linear_regression': LinearRegression(),
            'logistic_regression': LogisticRegression(),
            'kmeans': KMeans()
        }
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str, 
                   test_size: float = 0.2) -> Dict:
        """Train a machine learning model"""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Get and train the model
            model = self.models[model_type]
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            results = {
                'model': model,
                'predictions': y_pred,
                'feature_importance': self._get_feature_importance(model, X.columns)
            }
            
            # Add appropriate metrics based on model type
            if model_type in ['random_forest_classifier', 'logistic_regression']:
                results['accuracy'] = accuracy_score(y_test, y_pred)
                results['classification_report'] = classification_report(y_test, y_pred)
            else:
                results['mse'] = mean_squared_error(y_test, y_pred)
                results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Extract feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, model.coef_))
        return {}

class APIHandler:
    """Handles API interactions with OpenAI GPT-4"""
    
    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        if not self.api_key:
            raise ValueError("API_KEY must be set in .env file")
        
        # Set the base URL and endpoint for OpenAI
        self.api_base_url = 'https://api.openai.com'
        self.default_endpoint = 'v1/chat/completions'
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    async def send_data_async(self, data: Dict[str, Union[pd.DataFrame, Dict]], prompt: str) -> Dict:
        """Send data analysis request to GPT-4"""
        # Format the data summary for the prompt
        data_summary = f"Data shape: {data['metadata']['original_shape']}\n"
        data_summary += f"Columns: {', '.join(data['processed_data'].columns)}\n"
        
        payload = {
            'model': 'gpt-4',  # Specify GPT-4
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a data analysis assistant. Analyze the following data and respond to the user\'s prompt.'
                },
                {
                    'role': 'user',
                    'content': f"Data Information:\n{data_summary}\n\nUser's Request: {prompt}"
                }
            ],
            'temperature': 0,
            'max_tokens': 8000
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base_url}/{self.default_endpoint}",
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"API request failed: {str(e)}")
                raise

    def send_data(self, data: Dict[str, Union[pd.DataFrame, Dict]], prompt: str) -> Dict:
        """Synchronous wrapper for send_data_async"""
        return asyncio.run(self.send_data_async(data, prompt))

class DataProcessor:
    """Main orchestrator for data processing and API integration"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        try:
            self.api_handler = APIHandler()
            self.has_api = True
        except ValueError:
            logger.warning("API credentials not found. API features will be disabled.")
            self.has_api = False
    
    def run(self):
        """Run the data processing pipeline"""
        try:
            # Add colorful welcome message
            print("\n\033[1;36m=== Data Analysis Pipeline ===\033[0m")
            print("\033[1;37mThis tool will help you analyze your data and provide insights.\033[0m\n")

            # Enhanced input prompts
            print("\033[1;33mData Source Configuration:\033[0m")
            data_path = input("\033[1m→ Enter the path to your data file:\033[0m ").strip()
            target_col = input("\033[1m→ Enter target column name (press Enter if none):\033[0m ").strip() or None
            
            # Loading message with animation
            print("\n\033[1;32m⏳ Loading data...\033[0m")
            df = self.data_loader.load_data(data_path)
            print(f"\033[1;32m✓ Data loaded successfully. Shape: {df.shape}\033[0m")
            
            # Preprocessing messages
            print("\n\033[1;33m⚙️  Preprocessing data...\033[0m")
            processed_df = self.preprocessor.process_data(df)
            processed_data = {
                'processed_data': processed_df,
                'metadata': {
                    'original_shape': df.shape,
                    'target_column': target_col,
                    'preprocessing_steps': ['missing_values', 'duplicates', 'encoding', 'scaling']
                }
            }
            print("\033[1;32m✓ Preprocessing complete.\033[0m")
            print("\n\033[1;36mPreprocessing metadata:\033[0m")
            print("\033[37m" + json.dumps(processed_data['metadata'], indent=2) + "\033[0m")
            
            # API section with enhanced prompts
            if self.has_api:
                print("\n\033[1;35m🤖 Using GPT-4 for analysis...\033[0m")
                prompt = input("\033[1m→ Enter your analysis request:\033[0m ").strip()
                print("\n\033[1;33m⏳ Sending data and prompt to GPT-4...\033[0m")
                response = self.api_handler.send_data(processed_data, prompt)
                
                if 'choices' in response and len(response['choices']) > 0:
                    analysis = response['choices'][0]['message']['content']
                    print("\n\033[1;36mGPT-4 Analysis:\033[0m")
                    print("\033[37m" + analysis + "\033[0m")
                else:
                    print("\n\033[1;31m❌ No analysis received from GPT-4\033[0m")
            else:
                print("\n\033[1;31m⚠️  API features are disabled due to missing credentials.\033[0m")
            
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            print(f"\n\033[1;31m❌ An error occurred: {str(e)}\033[0m")
            print("\033[1;31mCheck the logs for more details.\033[0m")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.run()