# Data Analysis Pipeline

This project provides a comprehensive pipeline for data analysis, including data loading, preprocessing, analysis, visualization, statistical testing, machine learning, and integration with OpenAI GPT-4 for insights. Below is a detailed explanation of the project's components and usage instructions.

---

## Table of Contents

1. [Features](#features)
2. [Setup Instructions](#setup-instructions)
3. [Usage Guide](#usage-guide)
4. [Code Structure](#code-structure)
5. [Key Classes and Functions](#key-classes-and-functions)
6. [Local Analysis Options](#local-analysis-options)
7. [API Integration](#api-integration)
8. [Example Execution](#example-execution)

---

## Features

- **Data Loading**: Supports CSV, JSON, Excel, and Parquet formats.
- **Preprocessing**: Handles missing values, encodes categorical variables, removes duplicates, and scales numeric data.
- **Data Analysis**: Generates basic statistics, identifies missing values, and computes correlations.
- **Visualization**: Creates distribution plots, box plots, and correlation heatmaps.
- **Statistical Testing**: Performs normality and correlation tests.
- **Machine Learning**: Supports classification and regression using Random Forest, Linear/Logistic Regression, and K-Means clustering.
- **OpenAI Integration**: Sends processed data and prompts to GPT-4 for insights.
- **Customizable**: Select specific analysis options based on needs.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- Virtual environment (optional but recommended)
- Required Python libraries

### Installation Steps

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```env
     API_KEY=your_openai_api_key
     ```

---

## Usage Guide

### Running the Script

1. Execute the script:

   ```bash
   python datasc.py
   ```

2. Provide the required inputs when prompted:

   - Path to the data file.
   - Target column name (if applicable).
   - API endpoint (if using GPT-4 integration).
   - Custom prompt for analysis.

### Input Example

```text
Enter the path to your data file: data.csv
Enter target column name (press Enter if none): target
Enter API endpoint: analysis/predict
Enter your custom prompt: Analyze this data for trends
```

---

## Code Structure

- **`datasc.py`**: Main script orchestrating the entire pipeline.
- **Classes**:
  - `DataLoader`: Handles file validation and data loading.
  - `DataPreprocessor`: Performs data cleaning and transformation.
  - `DataAnalyzer`: Computes basic statistics and correlations.
  - `DataVisualizer`: Generates visualizations.
  - `StatisticalAnalyzer`: Conducts statistical tests.
  - `MachineLearning`: Implements machine learning models for classification and regression.
  - `APIHandler`: Manages interaction with the OpenAI GPT-4 API.
- **`requirements.txt`**: Lists dependencies.

---

## Key Classes and Functions

### `DataLoader`

- **Methods**:
  - `validate_file(filepath)`: Checks if the file exists and is of a supported format.
  - `load_data(source, **kwargs)`: Loads data from various formats.

### `DataPreprocessor`

- **Methods**:
  - `process_data(df)`: Executes the preprocessing pipeline.
  - Handles missing values, duplicates, encoding, and scaling.

### `DataAnalyzer`

- **Methods**:
  - `analyze_data(df)`: Computes statistics and correlations.

### `DataVisualizer`

- **Methods**:
  - `create_visualizations(df)`: Creates histograms, box plots, and heatmaps.

### `StatisticalAnalyzer`

- **Methods**:
  - `perform_statistical_tests(df)`: Performs normality and correlation tests.

### `MachineLearning`

- **Methods**:
  - `train_model(X, y, model_type, test_size)`: Trains models and computes performance metrics.

### `APIHandler`

- **Methods**:
  - `send_data(data, prompt)`: Sends processed data and prompts to GPT-4 for analysis.

---

## Local Analysis Options

When prompted, select one or more of the following:

1. Basic Statistics (summary, missing values, correlations)
2. Data Visualizations (distributions, correlations, box plots)
3. Statistical Tests (normality, correlations)
4. Classification Analysis (requires target column)
5. Regression Analysis (requires target column)
6. All of the above

---

## API Integration

- **Setup**: Add your OpenAI API key to the `.env` file.
- **Usage**: Input a custom prompt and endpoint when prompted.
- **Functionality**: Sends data and prompt to GPT-4 for advanced insights.

---

## Example Execution

1. Start the script:

   ```bash
   python datasc.py
   ```

2. Input data file path, target column, and API prompt.

3. Review preprocessing metadata and select analysis options.

4. View results (statistics, visualizations, ML metrics) and any GPT-4 insights.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements
