import plotly.graph_objects as go
import numpy as np

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


