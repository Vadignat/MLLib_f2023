import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import pandas as pd


class Visualisation():
    """
        A class for visualizing results using plotly.

        This class provides methods for creating various visual representations,
        such as plots, charts, etc., based on the plotly library.


        TODO:
            - Write a function `compare_model_predictions` in the class `Visualisation` that takes in:
                  1. x_values: Array-like data for the x-axis - our inputs .
                  2. y_values_list: A list of array-like data containing predictions from different models.
                  3. y_actual: Array-like data containing actual y-values - our targets.
                  4. title: A string to be used as the plot title.
              The function should generate a plot comparing model predictions from gradient descent and normal equation methods against actual data.



        """

    @staticmethod
    def compare_model_predictions(x_values, y_values_list, y_actual, title, names=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=y_actual, mode='markers', name='Actual Data'))

        for i, y_pred in enumerate(y_values_list):
            name = names[i] if names else f'Model {i + 1} Prediction'
            fig.add_trace(go.Scatter(x=x_values, y=y_pred, mode='lines', name=name))

        fig.update_layout(title=title, xaxis_title='X Values', yaxis_title='Y Values')
        return fig

    @staticmethod
    def plot_precision_recall_curve(thresholds, precision_values, recall_values, accuracy_values, f1_score_values):
        data = {
            'Thresholds': thresholds,
            'Precision': precision_values,
            'Recall': recall_values,
            'Accuracy': accuracy_values,
            'F1 Score': f1_score_values
        }
        df = pd.DataFrame(data)

        fig = px.line(df, x='Recall', y='Precision', text='Thresholds', title='Precision-Recall Curve')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='top right')

        hover_data = [f'Threshold: {threshold:.3f}<br>Accuracy: {accuracy:.2f}<br>F1 Score: {f1_score:.2f}' for
                      threshold, accuracy, f1_score in zip(thresholds, accuracy_values, f1_score_values)]
        fig.add_trace(go.Scatter(x=df['Recall'], y=df['Precision'], mode='markers', text=hover_data, name='Metrics'))

        fig.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision',
            legend_title='Hover Data',
        )

        return fig

