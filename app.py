from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static'
app.config['CHART_FILENAME'] = 'chart.png'
app.config['CHART_HTML'] = 'chart.html'

def apply_plotly_styling(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgb(30,30,50)',
        font_color='white',
        title_font_color='cyan',
        legend_bgcolor='rgba(255,255,255,0.1)',
        legend_bordercolor='black',
        legend_borderwidth=1
    )
    fig.update_xaxes(showgrid=True, gridcolor='gray', zerolinecolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='gray', zerolinecolor='lightgray')
    return fig

def save_chart():
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['CHART_FILENAME'])
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def save_plotly_chart(fig):
    if PLOTLY_AVAILABLE:
        chart_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['CHART_HTML'])
        fig.write_html(chart_path, include_plotlyjs='cdn')
        return chart_path
    else:
        return save_chart()

def clean_price_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].str.contains(r'^\s*[\$\u20B9\u20AC]', na=False).any():
                try:
                    df[col] = df[col].replace('[\$\u20B9\u20AC,]', '', regex=True).astype(float)
                except:
                    pass
    return df

def generate_bar_chart(df, x_col, y_col, title):
    df_grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).reset_index()
    if PLOTLY_AVAILABLE:
        fig = px.bar(df_grouped, x=x_col, y=y_col, title=title,
                     color=y_col, color_continuous_scale='viridis')
        fig.update_layout(title_font_size=20, xaxis_title=x_col, yaxis_title=y_col, template='plotly_white', height=600)
        return save_plotly_chart(fig)
    else:
        plt.figure(figsize=(10,6))
        plt.bar(df_grouped[x_col], df_grouped[y_col])
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return save_chart()

def generate_line_chart(df, x_col, y_col, title):
    df_clean = df[[x_col, y_col]].dropna()
    try:
        df_clean[x_col] = pd.to_datetime(df_clean[x_col], errors='coerce')
        if df_clean[x_col].notna().any():
            df_grouped = df_clean.groupby(df_clean[x_col].dt.to_period('M'))[y_col].sum().reset_index()
            df_grouped[x_col] = df_grouped[x_col].astype(str)
        else:
            df_grouped = df_clean.groupby(x_col)[y_col].sum().reset_index()
    except:
        df_grouped = df_clean.groupby(x_col)[y_col].sum().reset_index()
    fig = px.line(df_grouped, x=x_col, y=y_col, title=title, markers=True, line_shape='linear')
    fig.update_layout(title_font_size=20, xaxis_title=x_col, yaxis_title=y_col, template='plotly_white', height=600)
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    return save_plotly_chart(fig)

def generate_pie_chart(df, category_col, title):
    df_grouped = df[category_col].value_counts().reset_index()
    df_grouped.columns = [category_col, 'count']
    fig = px.pie(df_grouped, values='count', names=category_col, title=title, color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(title_font_size=20, template='plotly_white', height=600)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return save_plotly_chart(fig)

def generate_scatter_chart(df, x_col, y_col, title):
    df_clean = df[[x_col, y_col]].dropna()
    fig = px.scatter(df_clean, x=x_col, y=y_col, title=title, opacity=0.7, color=y_col, color_continuous_scale='viridis')
    fig.update_layout(title_font_size=20, xaxis_title=x_col, yaxis_title=y_col, template='plotly_white', height=600)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
    return save_plotly_chart(fig)

def generate_histogram(df, column, title):
    df_clean = df[[column]].dropna()
    mean_val = df_clean[column].mean()
    median_val = df_clean[column].median()
    fig = px.histogram(df_clean, x=column, title=title, nbins=30, color_discrete_sequence=['skyblue'])
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_val:.2f}")
    fig.add_vline(x=median_val, line_dash="dash", line_color="green", annotation_text=f"Median: {median_val:.2f}")
    fig.update_layout(title_font_size=20, xaxis_title=column, yaxis_title='Frequency', template='plotly_white', height=600, showlegend=False)
    return save_plotly_chart(fig)

def find_column(df_columns, user_words, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    user_words = [w.lower() for w in user_words]
    for col in df_columns:
        if col in exclude_cols:
            continue
        if col.lower() in user_words:
            return col
    for col in df_columns:
        if col in exclude_cols:
            continue
        if any(word in col.lower() for word in user_words):
            return col
    return None

def find_best_columns_for_chart(df, chart_type, query_words):
    cols = df.columns.tolist()
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if df[c].dtype == 'object' or df[c].nunique() < 20]
    date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in c.lower() for k in ['date', 'time', 'year', 'month'])]
    if chart_type == 'scatter':
        if len(numeric_cols) >= 2:
            x_col = find_column(numeric_cols, query_words)
            y_col = find_column(numeric_cols, query_words, exclude_cols=[x_col] if x_col else [])
            if not x_col:
                x_col = numeric_cols[0]
            if not y_col:
                y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            return x_col, y_col
        else:
            return None, None
    elif chart_type == 'line':
        x_col = find_column(date_cols, query_words) or find_column(categorical_cols, query_words) or (date_cols[0] if date_cols else categorical_cols[0] if categorical_cols else cols[0])
        y_col = find_column(numeric_cols, query_words) or (numeric_cols[0] if numeric_cols else None)
        return x_col, y_col
    elif chart_type in ['bar', 'pie']:
        x_col = find_column(categorical_cols, query_words) or (categorical_cols[0] if categorical_cols else cols[0])
        y_col = find_column(numeric_cols, query_words) or (numeric_cols[0] if numeric_cols else None)
        return x_col, y_col
    elif chart_type == 'hist':
        hist_col = find_column(numeric_cols, query_words) or (numeric_cols[0] if numeric_cols else None)
        return hist_col, None
    return None, None

def handle_query(df, query):
    query_lower = query.lower()
    words = query_lower.split()
    chart_types = ['scatter', 'line', 'bar', 'pie', 'hist', 'histogram']
    chart_type = next((c for c in chart_types if c in query_lower), None)
    if not chart_type:
        return None, "No valid chart type found."
    chart_type = 'hist' if chart_type in ['hist', 'histogram'] else chart_type
    x_col, y_col = find_best_columns_for_chart(df, chart_type, words)
    if chart_type == 'hist' and not x_col:
        return None, "Could not find numeric column for histogram."
    if chart_type != 'hist' and (not x_col or not y_col):
        return None, f"Could not find suitable columns for {chart_type} chart."
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
        return None, f"Error generating chart: {str(e)}"

def validate_dataframe(df):
    if df.empty:
        return False, "Uploaded file is empty."
    if len(df.columns) < 2:
        return False, "File must have at least 2 columns."
    if len(df) < 2:
        return False, "File must have at least 2 rows."
    return True, None

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_path, message, columns, data_info = None, None, None, None
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            query = request.form.get('query', '').strip()
            if not file or file.filename == '':
                return render_template('index.html', message="Please upload a file.")
            if not query:
                return render_template('index.html', message="Please enter a query.")
            if not (file.filename.lower().endswith('.csv') or file.filename.lower().endswith('.xlsx')):
                return render_template('index.html', message="Only CSV or Excel files are supported.")
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            df = pd.read_csv(file_path) if file.filename.endswith('.csv') else pd.read_excel(file_path)
            df = clean_price_columns(df)
            is_valid, validation_error = validate_dataframe(df)
            if not is_valid:
                return render_template('index.html', message=validation_error)
            columns = df.columns.tolist()
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            categorical_cols = [c for c in columns if df[c].dtype == 'object' or df[c].nunique() < 20]
            data_info = {
                'total_rows': len(df),
                'total_columns': len(columns),
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols
            }
            chart_path, message = handle_query(df, query)
            os.remove(file_path)
        except pd.errors.EmptyDataError:
            message = "The uploaded file is empty or corrupted."
        except pd.errors.ParserError:
            message = "Could not parse the file. Please check the format."
        except Exception as e:
            message = f"An error occurred: {str(e)}"
    return render_template('index.html', chart_path=chart_path, message=message, columns=columns, data_info=data_info)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)
