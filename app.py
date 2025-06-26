#  This function is used to display HTML files (your web pages) to the user.                       RENDER TEMPLATES
#  It can also pass data from your Python code to these HTML templates.

# This object allows your Flask application to access incoming data from web requests,             REQUEST
# such as files uploaded by a user or data submitted through a form.

# The os module provides functions for interacting with the operating system.
# Here, it's used to manage file paths,                                                            OS   
# ensuring that the application can correctly locate its template and static folders,
# and save uploaded files and generated charts regardless of the operating system it's running on.

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# Try to import Plotly, fall back to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
    print("Plotly imported successfully")
except ImportError as e:
    print(f"Plotly not available: {e}")
    PLOTLY_AVAILABLE = False

# Get the directory where this script is located
try:
    basedir = os.path.abspath(os.path.dirname(__file__))
except NameError:
    basedir = os.path.abspath(os.path.dirname('app.py'))

# Configure Flask with explicit paths
app = Flask(__name__,
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))

app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static')
app.config['CHART_FILENAME'] = 'chart.png'
app.config['CHART_HTML'] = 'chart.html'


# The join() method in Python is a string method that provides a flexible way to 
# concatenate elements of an iterable (like a list, tuple, or set) into a single string.            JOIN
# It's called on the delimiter string that you want to use to join the elements.


def save_chart():
    """Save matplotlib chart"""
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['CHART_FILENAME'])
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def save_plotly_chart(fig):
    """Save Plotly chart as HTML"""
    if PLOTLY_AVAILABLE:
        chart_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['CHART_HTML'])
        fig.write_html(chart_path, include_plotlyjs='cdn')
        return chart_path
    else:
        return save_chart()  # Fallback to matplotlib

def generate_bar_chart(df, x_col, y_col, title):
    """Generate bar chart using Plotly or matplotlib"""
    df_grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).reset_index()

    if PLOTLY_AVAILABLE:
        # Use Plotly for interactive chart
        fig = px.bar(df_grouped, x=x_col, y=y_col, title=title,
                     color=y_col, color_continuous_scale='viridis')

        fig.update_layout(
            title_font_size=20,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white',
            height=600
        )

        return save_plotly_chart(fig)
    else:
        # Fallback to matplotlib
        plt.figure(figsize=(10,6))
        plt.bar(df_grouped[x_col], df_grouped[y_col])
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return save_chart()

def generate_line_chart(df, x_col, y_col, title):
    """Generate interactive line chart using Plotly"""
    # Clean data
    df_clean = df[[x_col, y_col]].dropna()

    # Try to convert x_col to datetime if it looks like dates
    try:
        df_clean[x_col] = pd.to_datetime(df_clean[x_col], errors='coerce')
        # If successful, group by date periods
        if df_clean[x_col].notna().any():
            df_grouped = df_clean.groupby(df_clean[x_col].dt.to_period('M'))[y_col].sum().reset_index()
            df_grouped[x_col] = df_grouped[x_col].astype(str)
        else:
            # If datetime conversion failed, use original data
            df_grouped = df_clean.groupby(x_col)[y_col].sum().reset_index()
    except:
        # If datetime conversion fails, group by categorical values
        df_grouped = df_clean.groupby(x_col)[y_col].sum().reset_index()

    fig = px.line(df_grouped, x=x_col, y=y_col, title=title,
                  markers=True, line_shape='linear')

    fig.update_layout(
        title_font_size=20,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=600
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=8))

    return save_plotly_chart(fig)

def generate_pie_chart(df, category_col, title):
    """Generate interactive pie chart using Plotly"""
    df_grouped = df[category_col].value_counts().reset_index()
    df_grouped.columns = [category_col, 'count']

    fig = px.pie(df_grouped, values='count', names=category_col, title=title,
                 color_discrete_sequence=px.colors.qualitative.Set3)

    fig.update_layout(
        title_font_size=20,
        template='plotly_white',
        height=600
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')

    return save_plotly_chart(fig)

def generate_scatter_chart(df, x_col, y_col, title):
    """Generate interactive scatter plot using Plotly"""
    # Clean data - remove rows where either column has NaN
    df_clean = df[[x_col, y_col]].dropna()

    fig = px.scatter(df_clean, x=x_col, y=y_col, title=title,
                     opacity=0.7, color=y_col, color_continuous_scale='viridis')

    fig.update_layout(
        title_font_size=20,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=600
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))

    return save_plotly_chart(fig)

def generate_histogram(df, column, title):
    """Generate interactive histogram using Plotly"""
    if PLOTLY_AVAILABLE:
        # Clean data - remove NaN values
        df_clean = df[[column]].dropna()

        # Calculate statistics
        mean_val = df_clean[column].mean()
        median_val = df_clean[column].median()

        # Create histogram using DataFrame
        fig = px.histogram(df_clean, x=column, title=title, nbins=30,
                           color_discrete_sequence=['skyblue'])

        # Add mean and median lines
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                      annotation_text=f"Median: {median_val:.2f}")

        fig.update_layout(
            title_font_size=20,
            xaxis_title=column,
            yaxis_title='Frequency',
            template='plotly_white',
            height=600,
            showlegend=False
        )

        return save_plotly_chart(fig)
    else:
        # Fallback to matplotlib
        data = df[column].dropna()

        plt.figure(figsize=(10,6))
        plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(title, fontsize=16)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        plt.legend()
        plt.tight_layout()
        return save_chart()

def find_column(df_columns, user_words, exclude_cols=None):
    """Find a column that matches any of the user words"""
    if exclude_cols is None:
        exclude_cols = []

    user_words = [w.lower() for w in user_words]

    # First try exact matches
    for col in df_columns:
        if col in exclude_cols:
            continue
        col_lower = col.lower()
        if col_lower in user_words:
            return col

    # Then try partial matches
    for col in df_columns:
        if col in exclude_cols:
            continue
        col_lower = col.lower()
        if any(word in col_lower for word in user_words):
            return col
    return None

def find_best_columns_for_chart(df, chart_type, query_words):
    """Smart column selection based on chart type and data types"""
    cols = df.columns.tolist()
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if df[c].dtype == 'object' or df[c].nunique() < 20]
    date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c]) or
                 any(keyword in c.lower() for keyword in ['date', 'time', 'year', 'month'])]

    if chart_type == 'scatter':
        # For scatter plots, we need two numeric columns
        if len(numeric_cols) >= 2:
            # Try to find columns mentioned in query
            x_col = find_column(numeric_cols, query_words)
            y_col = find_column(numeric_cols, query_words, exclude_cols=[x_col] if x_col else [])

            # If not found in query, use first two numeric columns
            if not x_col:
                x_col = numeric_cols[0]
            if not y_col:
                y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

            return x_col, y_col
        else:
            return None, None

    elif chart_type == 'line':
        # For line charts, prefer date/categorical for x-axis and numeric for y-axis
        x_col = None
        y_col = None

        # Try to find columns mentioned in query
        if date_cols:
            x_col = find_column(date_cols, query_words)
        if not x_col and categorical_cols:
            x_col = find_column(categorical_cols, query_words)
        if not x_col:
            # Default to first date or categorical column
            x_col = date_cols[0] if date_cols else (categorical_cols[0] if categorical_cols else cols[0])

        if numeric_cols:
            y_col = find_column(numeric_cols, query_words)
            if not y_col:
                y_col = numeric_cols[0]

        return x_col, y_col

    elif chart_type in ['bar', 'pie']:
        # For bar/pie charts, categorical for x-axis and numeric for y-axis
        x_col = None
        y_col = None

        if categorical_cols:
            x_col = find_column(categorical_cols, query_words)
            if not x_col:
                x_col = categorical_cols[0]

        if numeric_cols:
            y_col = find_column(numeric_cols, query_words)
            if not y_col:
                y_col = numeric_cols[0]

        return x_col, y_col

    elif chart_type == 'hist':
        # For histograms, we only need one numeric column
        if numeric_cols:
            hist_col = find_column(numeric_cols, query_words)
            if not hist_col:
                hist_col = numeric_cols[0]
            return hist_col, None
        else:
            return None, None

    return None, None

def handle_query(df, query):
    """Enhanced query handler with smart column selection"""
    query_lower = query.lower()
    words = query_lower.split()

    # Detect chart type
    chart_types = ['scatter', 'line', 'bar', 'pie', 'hist', 'histogram']  # Order matters for detection
    chart_type = None
    for ctype in chart_types:
        if ctype in query_lower:
            chart_type = 'hist' if ctype in ['hist', 'histogram'] else ctype
            break

    if not chart_type:
        return None, "No valid chart type found in query. Please mention one of: bar, line, pie, scatter, hist/histogram."

    # Use smart column selection
    x_col, y_col = find_best_columns_for_chart(df, chart_type, words)

    # For histograms, only x_col is needed (y_col will be None)
    if chart_type == 'hist':
        if not x_col:
            return None, f"Could not find suitable numeric column for histogram. Please check your data."
    else:
        if not x_col or not y_col:
            return None, f"Could not find suitable columns for {chart_type} chart. Please check your data."

    # Generate appropriate chart
    try:
        if chart_type == 'scatter':
            return generate_scatter_chart(df, x_col, y_col, f'Scatter plot: {x_col} vs {y_col}'), None
        elif chart_type == 'line':
            return generate_line_chart(df, x_col, y_col, f'Line chart: {y_col} over {x_col}'), None
        elif chart_type == 'bar':
            return generate_bar_chart(df, x_col, y_col, f'Bar chart: {y_col} by {x_col}'), None
        elif chart_type == 'pie':
            return generate_pie_chart(df, x_col, f'Pie chart: {x_col} distribution'), None
        elif chart_type == 'hist':
            return generate_histogram(df, x_col, f'Histogram: {x_col} distribution'), None
    except Exception as e:
        return None, f"Error generating {chart_type} chart: {str(e)}"

    return None, "Unable to process your query with this data."

def validate_dataframe(df):
    """Validate uploaded dataframe"""
    if df.empty:
        return False, "The uploaded file is empty."

    if len(df.columns) < 2:
        return False, "The file must have at least 2 columns."

    if len(df) < 2:
        return False, "The file must have at least 2 rows of data."

    return True, None

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_path = None
    message = None
    columns = None
    data_info = None

    if request.method == 'POST':
        try:
            file = request.files.get('file')
            query = request.form.get('query', '').strip()

            # Validate inputs
            if not file or file.filename == '':
                message = "Please upload a CSV file."
                return render_template('index.html', message=message)

            if not query:
                message = "Please enter a query describing the chart you want."
                return render_template('index.html', message=message)

            # Validate file type
            if not file.filename.lower().endswith('.csv'):
                message = "Please upload a CSV file only."
                return render_template('index.html', message=message)

            # Save and read file
            import uuid
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read and validate dataframe
            df = pd.read_csv(file_path)
            is_valid, validation_error = validate_dataframe(df)

            if not is_valid:
                message = validation_error
                return render_template('index.html', message=message)

            columns = df.columns.tolist()

            # Create data info for user
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            categorical_cols = [c for c in columns if df[c].dtype == 'object' or df[c].nunique() < 20]

            data_info = {
                'total_rows': len(df),
                'total_columns': len(columns),
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols
            }

            # Generate chart
            chart_path, message = handle_query(df, query)

            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass

        except pd.errors.EmptyDataError:
            message = "The uploaded file is empty or corrupted."
        except pd.errors.ParserError:
            message = "Could not parse the CSV file. Please check the file format."
        except Exception as e:
            message = f"An error occurred: {str(e)}"

    return render_template('index.html',
                         chart_path=chart_path,
                         message=message,
                         columns=columns,
                         data_info=data_info)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='127.0.0.1', port=5002)
