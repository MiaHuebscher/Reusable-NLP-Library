'''
Mia Huebscher
Assignment 1

A python file containing reusable functions for the construction of Sankey diagrams
'''
# Import library
import plotly.graph_objects as go

def add_vals_col(df, threshold=22):
    '''
    Adds a value column that illustrates the frequency counts of links

    :param df: a Dataframe without a value column
    :param threshold: the minimum number of link occurrences needed for the link to be graphed
    :return df: a Dataframe with a value column
    '''
    # Add value colum
    df = df.groupby(df.columns.to_list(), as_index=False).size()
    df = df.rename(columns={'size':'Value'})

    # Filter out rows with values under some threshold
    df = df[df.Value > threshold]

    return df

def code_mapping(df, src, targ):
    '''
    Maps labels in src and targ columns to integers

    :param df: A Dataframe containing at least a source and target column
    :param src: The column containing labels you want for the source nodes
    :param targ: The column containing labels you want for the target nodes
    :return:
    '''
    # Extract distinct labels for the Sankey nodes
    labels = list(df[src]) + list(df[targ])

    # Define a list of integer codes
    codes = list(range(len(labels)))

    # Create a dictionary mapping the labels to random codes
    lc_map = dict(zip(labels, codes))

    # In the df, substitute the codes for the labels
    df = df.replace({src: lc_map, targ: lc_map})

    return df, labels

def make_sankey(df, src, targ, title, vals=None, **kwargs):
    '''
    Generates the sankey diagram

    :param df: a Dataframe containing the data needed for the Sankey diagram
    :param src: the column name that has the data for the source nodes
    :param targ: the column name that has the data for the target nodes
    :param title: the title for the Sankey diagram
    :param vals: the frequency counts of links
    :param kwargs: any other arguments needed to customize the Sankey diagram
    :return: the Sankey diagram
    '''
    # Call the code_mapping function to get a revised Dataframe and labels
    df, labels = code_mapping(df, src, targ)

    # Retrieve the 'Value' data
    if vals:
        values = df[vals]

    # If 'Value' is not given as a parameter, set all values to zero
    else:
        values = [1] * len(df)

    # Customize the Sankey diagram
    pad = kwargs.get('pad', 10)
    thickness = kwargs.get('thickness', 70)
    line_color = kwargs.get('line_color', 'black')
    line_width = kwargs.get('line_width', 1)

    # Create the links and nodes for the Sankey diagram
    link = {'source': df[src], 'target': df[targ], 'value': values}
    node = {'label': labels, 'pad':pad, 'thickness':thickness, 'line':{'color':line_color, 'width':line_width}}

    # Graph the Sankey Diagram
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    fig.update_layout(title_text=title, font_size=10)
    fig.show()