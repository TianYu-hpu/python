import pandas as pd
import numpy as np
import bokeh
from bokeh.plotting import figure, gridplot
from bokeh.io import output_notebook, show
from bokeh.palettes import brewer, Category20, Oranges, Reds, Purples
from bokeh.layouts import layout, grid, column, row
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import factor_cmap, linear_cmap, LinearColorMapper

output_notebook()

class PLOT_TYPE():
    BAR          =  1
    BOX          =  2
    HIST         =  3
    HIST_BY_CATE =  4
    SCATTER      =  5
    CORR         =  6
    NONE         =  0

class PLOT_STYLE():
    FRAME_WIDTH = 132
    FRAME_HEIGHT = 132
    MIN_BORDER_LEFT = 50
    PLOT_PADDING = 5
    TOOLS = None
    
def get_plot_type(df, row_variable, col_variable, row_idx, col_idx):
    dtype_row = df[row_variable].describe().dtype.name
    dtype_col = df[col_variable].describe().dtype.name
    
    if row_variable == col_variable:
        if dtype_row in ['object']:
            return PLOT_TYPE.BAR
        if dtype_row in ['float64']:
            return PLOT_TYPE.HIST
    else:
        if dtype_row in ['object'] and dtype_col in ['float64']:
            return PLOT_TYPE.BOX
        if dtype_row in ['float64'] and dtype_col in ['object']:
            return PLOT_TYPE.HIST_BY_CATE
        if dtype_row in ['float64'] and dtype_col in ['float64']:
            if row_idx > col_idx:
                return PLOT_TYPE.SCATTER
            else:
                return PLOT_TYPE.CORR
        
    return PLOT_TYPE.NONE

def vbar_plot(df, cat, colors):
    df_count = df.groupby(by=cat).agg({cat: 'count'})
    data = {
        'x': df_count.index.tolist(),
        'top': df_count[cat], 
        'colors': colors[len(df_count)]
    }
    p = figure(
        x_range      = data['x'],
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
        # x_axis_location='above'
    )
    p.vbar(x='x', top='top', width=0.9, fill_color='colors', line_width=0, source=data)
    p.y_range.start = 0
    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    return p

def generate_box_plot_data(df, category, column):

    # find the quartiles and IQR for each category
    groups = df[[column, category]].groupby(by=category, sort=False)
    # generate some synthetic time series for six different categories
    cats = []

    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    # find the outliers for each category
    def outliers(group):
        cat = group.name
        cats.append(cat)
        return group[(group[column] > upper.loc[cat][column]) | (group[column] < lower.loc[cat][column])][column]
    out = groups.apply(outliers).dropna()
    # prepare outlier data for plotting, we need coordinates for every outlier.
    out_index = []
    out_category = []
    out_value = []
    if not out.empty:
        for keys in out.index:
            out_index.append(keys[1])
            out_category.append(keys[0])
            out_value.append(out.loc[keys[0]].loc[keys[1]])

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)

    upper[column] = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,column]),upper[column])]
    lower[column] = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,column]),lower[column])]

    data_box_plot = {
        'category': cats,
        'qmin': qmin[column].tolist(),
        'q1' : q1[column].tolist(),
        'q2' : q2[column].tolist(),
        'q3' : q3[column].tolist(),
        'iqr': iqr[column].tolist(),
        'qmax': qmax[column].tolist(),
        'upper': upper[column].tolist(),
        'lower': lower[column].tolist()
    }

    data_box_plot_outlier = {
        'index': out_index,
        'category' : out_category,
        'value': out_value
    }
    
    return data_box_plot, data_box_plot_outlier

def box_plot(df, cat, cont, colors):
    
    categories = [by for by, group in df.groupby(by=cat)]
    colors_cat = colors[len(categories)]
    box, out = generate_box_plot_data(df, cat, cont)
    p = figure(
        y_range      = categories[::-1],
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
    )
    
    box = ColumnDataSource(data=box)
    # stems
    p.segment(x0='upper', y0='category', x1='q3', y1='category', line_color='black', source=box)
    p.segment(x0='lower', y0='category', x1='q1', y1='category', line_color='black', source=box)

    # boxes
    p.hbar(y='category', height=0.7, right='q2', left='q3', fill_color='white', fill_alpha=0.5, 
           line_color=factor_cmap('category', colors_cat, categories), line_width=2, 
           source=box)
    p.hbar(y='category', height=0.7, right='q1', left='q2', fill_color='white', fill_alpha=0.3, 
           line_color=factor_cmap('category', colors_cat, categories), line_width=2, 
           source=box)

    # whiskers (almost-0 height rects simpler than segments)
    p.dash(x='lower', y='category', size=10, angle=np.pi/2, line_width=1.5, line_color='black', source=box)
    p.dash(x='upper', y='category', size=10, angle=np.pi/2, line_width=1.5, line_color='black', source=box)

    # outliers
    p.circle(x='value', y='category', size=4, color="#F38630", fill_alpha=1, source=out)

    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    return p

def hist_plot(df, cont):
    p = figure(
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
    )
    hist, edges = np.histogram(
        df[cont], 
        # bins = 11,
        # range=(0, df[cont].max())
    )
    data = {
        'hist': hist,
        'left': edges[:-1],
        'right': edges[1:]
    }
    p.quad(
        top='hist', bottom=0, left='left', right='right',
        fill_color='dimgray', line_width=0,
        source=data
    )
    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    
    return p

def scatter_plot(df, cat, cont_1, cont_2, colors):
    
    colors_cat = colors[len(df[cat].unique())]
    p = figure(
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
    )
    
    for i, (by, group) in enumerate(df.groupby(by=cat)):
        
        data = {
            'x': group[cont_1],
            'y': group[cont_2]
        } 
        p.scatter(x='x', y='y', color=colors_cat[i], alpha=0.5, source=data)
        
    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    
    return p
    
def corr_plot(df, cont_1, cont_2):

    corr = df[[cont_1, cont_2]].corr()
    
    p = figure(
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
    )

    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    p.text(x=0, y=0, y_offset=-10, text=['Corr:'], text_baseline='middle', align='center')
    p.text(x=0, y=0, y_offset=10, text=['%.3f' % corr.loc[cont_1, cont_2]], text_baseline='middle', align='center')
    return p

def label_plot(text, orien='h'):
    
    p = None
    
    if orien == 'h':
        p = figure(
            frame_width  = PLOT_STYLE.FRAME_WIDTH,
            frame_height = PLOT_STYLE.FRAME_HEIGHT // 4,
            tools=''
        )
        p.text(x=0, y=0, text=[text], text_baseline='middle', align='center', text_color='white')
        p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
        p.min_border_bottom = 0
    
    if orien == 'v':
        p = figure(
            frame_width  = PLOT_STYLE.FRAME_WIDTH // 4,
            frame_height = PLOT_STYLE.FRAME_HEIGHT,
            tools=''
        )
        p.text(x=0, y=0, text=[text], text_baseline='middle', align='center', text_color='white', angle=3*np.pi/2)
        p.min_border_left = 0
    
    p.background_fill_color = 'deepskyblue'
    p.grid.visible = False
    p.axis.visible = False
    p.toolbar.logo = None
    return p

def legend_plot(categories, variables, colors):
    
    colors_cat = colors[len(categories)][::-1]
    size = 10
    times = 20
    padding = 2
    frame_height = round(PLOT_STYLE.FRAME_HEIGHT * len(variables) + 
                         PLOT_STYLE.FRAME_HEIGHT // 4 + 
                         len(variables) * PLOT_STYLE.PLOT_PADDING * 2)
    p = figure(
        tools         = '', 
        frame_width   = size * times,
        frame_height  = frame_height,
        x_range       = [0, size * times],
        y_range       = [-(frame_height - PLOT_STYLE.FRAME_HEIGHT // 4 - PLOT_STYLE.PLOT_PADDING * 2) / 2, 
                          (frame_height - PLOT_STYLE.FRAME_HEIGHT // 4 - PLOT_STYLE.PLOT_PADDING * 2) / 2 + (PLOT_STYLE.FRAME_HEIGHT // 4 + PLOT_STYLE.PLOT_PADDING * 2)]
    )
    
    step = 0
    if len(categories) % 2 == 0:
        step = -size * (len(categories) / 2 + 1)
    else:
        step = -size * (len(categories) / 2 + 1.5)
        
    for i, c in enumerate(categories[::-1]):
        p.square(x=size, y=step, size=size, color=colors_cat[i])
        p.text(x=size, y=step, text=[c], text_font_size={'value': '10pt'}, text_baseline='middle', align='left', x_offset=size)
        step += size * padding

    p.axis.visible = False
    p.grid.visible = False
    p.outline_line_width = 0
    p.min_border_left = 0
    p.x_range.start = 0
    p.toolbar.logo = None
    
    return p

def decorate_plot_axis(p, length, i, j):

    # 如果是第一行标签
    if i == -1:
        if j != 0:
            p.min_border_left = PLOT_STYLE.PLOT_PADDING
            p.min_border_right = PLOT_STYLE.PLOT_PADDING
            p.min_border_top = PLOT_STYLE.PLOT_PADDING
            p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        else:
            p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT    
            p.min_border_right = PLOT_STYLE.PLOT_PADDING
            p.min_border_top = PLOT_STYLE.PLOT_PADDING
            p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        return
    
    # 如果是最右面的标签列
    if j == -1:
        p.min_border_left = PLOT_STYLE.PLOT_PADDING
        p.min_border_right = PLOT_STYLE.PLOT_PADDING
        p.min_border_top = PLOT_STYLE.PLOT_PADDING
        p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        return
    
    # 不是最后一行也不是第一列
    if length - 1 != i and j != 0:
        if type(p) == bokeh.plotting.Figure:
            p.axis.visible = False
            p.min_border_left = PLOT_STYLE.PLOT_PADDING
            p.min_border_right = PLOT_STYLE.PLOT_PADDING
            p.min_border_top = PLOT_STYLE.PLOT_PADDING
            p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        if type(p) == bokeh.models.layouts.GridBox:
            for p,_,_ in p.children:
                p.axis.visible = False
                p.min_border_left = PLOT_STYLE.PLOT_PADDING
                p.min_border_right = PLOT_STYLE.PLOT_PADDING
                p.min_border_top = PLOT_STYLE.PLOT_PADDING
                p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        return
    
    # 如果是第一列
    if j == 0:
        # 如果不是最后一行
        if i != length - 1:
            if type(p) == bokeh.plotting.Figure:
                p.xaxis.visible = False
                p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
                p.min_border_right = PLOT_STYLE.PLOT_PADDING
                p.min_border_top = PLOT_STYLE.PLOT_PADDING
                p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
            if type(p) == bokeh.models.layouts.GridBox:
                p.children[-1][0].min_border_right = PLOT_STYLE.PLOT_PADDING
                for p,_,_ in p.children:
                    p.min_border_top = PLOT_STYLE.PLOT_PADDING
                    p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
                    p.xaxis.visible = False
        # 如果是最后一行，也就是左下角的绘图
        else:
            if type(p) == bokeh.plotting.Figure:
                p.xaxis.visible = False
                p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
                p.min_border_right = PLOT_STYLE.PLOT_PADDING
                p.min_border_top = PLOT_STYLE.PLOT_PADDING
                p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
            if type(p) == bokeh.models.layouts.GridBox:
                p.children[-1][0].min_border_right = PLOT_STYLE.PLOT_PADDING
                for p,_,_ in p.children:
                    p.min_border_top = PLOT_STYLE.PLOT_PADDING
                    p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
                    p.xaxis.major_label_orientation = np.pi/2
                    p.xaxis.ticker.desired_num_ticks = 3
    # 如果不是第一列，那么这里就是最后一行
    else:
        if type(p) == bokeh.plotting.Figure:
            p.yaxis.visible = False
            p.min_border_left = PLOT_STYLE.PLOT_PADDING
            p.min_border_right = PLOT_STYLE.PLOT_PADDING
            p.min_border_top = PLOT_STYLE.PLOT_PADDING
            p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        if type(p) == bokeh.models.layouts.GridBox:
            p.children[-1][0].min_border_right = PLOT_STYLE.PLOT_PADDING
            p.children[0][0].min_border_left = PLOT_STYLE.PLOT_PADDING
            for p,_,_ in p.children:
                p.min_border_top = PLOT_STYLE.PLOT_PADDING
                p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
                p.xaxis.major_label_orientation = np.pi/2
                p.xaxis.ticker.desired_num_ticks = 3

def hist_by_cate_plot(df, cat, cont, colors):
    plots = []
    cat_length = len(df[cat].unique())
    cat_colors = colors[cat_length]
    
    x_max = 0
    for i, (by, group) in enumerate(df.groupby(by=cat)):
        p = figure(
            frame_width  = (PLOT_STYLE.FRAME_WIDTH - (cat_length - 1) * 2 * 2) // cat_length,
            frame_height = PLOT_STYLE.FRAME_HEIGHT,
            tools='save'
        )
        hist, edges = np.histogram(
            group[cont], 
            range=(0, df[cont].max())
        )
        hist_max = max(hist)
        if x_max < hist_max: x_max = hist_max
        data = {
            'hist': hist,
            'bottom': edges[:-1],
            'top': edges[1:]
        }
        p.quad(
            top='top', bottom='bottom', left=0, right='hist',
            fill_color=cat_colors[i], line_width=0,
            source=data
        )
        plots.append(p)

    for i, p in enumerate(plots):
        if i == 0: 
            p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
            p.min_border_right = 2
        else:
            p.min_border_left  = 2
            p.min_border_right = 2
            p.yaxis.visible = False

        p.xaxis.major_label_orientation = np.pi/2
        p.xaxis.ticker.desired_num_ticks = 3
        p.xaxis.ticker.num_minor_ticks = 0
        p.x_range.start = -x_max * 0.05
        p.x_range.end = x_max * 1.05
    
    grid = gridplot(plots, ncols=cat_length, toolbar_location=None)
    return grid

def pair_plot(df, variables):
    plots = np.full(
        [len(variables)+1, len(variables)+1], 
        None
    )
    # 行数据类型
    for i, _v_row in enumerate(variables):
        # 列数据类型
        for j, _v_col in enumerate(variables):
            plot_type = get_plot_type(df, _v_row, _v_col, i, j)
            p = None
            if plot_type == PLOT_TYPE.BAR:
                p = vbar_plot(df, _v_row, brewer['Set1'])
            if plot_type == PLOT_TYPE.BOX:
                p = box_plot(df, _v_row, _v_col, brewer['Set1'])
            if plot_type == PLOT_TYPE.HIST:
                p = hist_plot(df, _v_row)
            if plot_type == PLOT_TYPE.HIST_BY_CATE:
                p = hist_by_cate_plot(df, _v_col, _v_row, brewer['Set1'])
            if plot_type == PLOT_TYPE.SCATTER:
                p = scatter_plot(df, variables[0], _v_col, _v_row, brewer['Set1'])
            if plot_type == PLOT_TYPE.CORR:
                p = corr_plot(df, _v_col, _v_row)
            
            decorate_plot_axis(p, len(variables), i, j)
            plots[i+1, j] = p
    
    for i, v in enumerate(variables):
        p = label_plot(v, orien='h')
        decorate_plot_axis(p, len(variables), -1, i)
        plots[0, i] = p

    for i, v in enumerate(variables):
        p = label_plot(v, orien='v')
        decorate_plot_axis(p, len(variables), i, -1)
        plots[i+1, len(variables)] = p

    plots = plots.reshape(-1).tolist()

    grid = gridplot(plots, ncols=len(variables)+1, toolbar_location=None)
    p = legend_plot([by for by, group in df.groupby(by=variables[0])], variables, brewer['Set1'])
    
    return row(grid, p, sizing_mode='stretch_height')

def heatmap_plot(df, row_variables, col_variables, full=False):
    
    corr_array = np.full(
        (len(row_variables), len(row_variables)), 
        .0
    )
    for i in range(len(row_variables)):
        for j in range(len(col_variables)):
            corr = df[[row_variables[i], col_variables[j]]].corr()
            corr_array[i, j] = corr.iloc[0, 1]
            
    p = figure(
        x_range = col_variables,
        y_range = row_variables,
        frame_width = 400, frame_height = 400
    )

    row_num = corr_array.shape[0]
    col_num = corr_array.shape[1]
    
    if full == False:
        data = {
            'x': [_r + 0.5 for _c in range(col_num) for _r in range(_c + 1)],
            'y': [_c + 0.5 for _c in range(col_num) for _r in range(_c + 1)],
            'corr': [corr_array.reshape(-1)[_c * col_num + _r] for _c in range(col_num) for _r in range(_c + 1)]
        }
    else:
        data = {
            'x': [_c + 0.5 for _c in range(col_num)] * row_num,
            'y': [_p // col_num + 0.5 for _p in range(col_num * row_num)],
            'corr': corr_array.reshape(-1).tolist()
        }

    data['corr_text'] = ['%.2f' % _corr for _corr in data['corr']]
    
    palette = Purples[256] + Oranges[256][::-1]
    low = min(data['corr'])
    high = max(data['corr'])
    palette = palette[int((low - (-1)) / 2 * 512): int((high - (-1)) / 2 * 512)]
    alpha = 1.
    
    p.rect(
        x='x', y='y',
        width=1, height=1, 
        line_color='dimgrey', line_width=0.5, line_alpha=0.5,
        fill_color=linear_cmap('corr', palette, low=low, high=high),
        fill_alpha=alpha,
        source=data
    )

    p.text(
        x='x', y='y',
        text='corr_text', text_font_size={'value': '12px'},
        text_baseline='middle', align='center', 
        source=data
    )
    
    color_bar = ColorBar(
        color_mapper=LinearColorMapper(low=low, high=high, palette=palette),
        major_label_text_font_style='bold',
        major_tick_line_color='black',
        label_standoff=6, 
        border_line_color=None,
        location=(0,0)
    )
    p.add_layout(color_bar, 'right')

    p.x_range.range_padding = 0.03
    p.y_range.range_padding = 0.03
    p.xaxis.major_label_orientation = .8
    p.axis.axis_label_text_font_size = '14pt'
    p.axis.major_label_text_font_size = '10pt'
    
    return p

def parallel_plot(df, cat, variables):

    df = df.copy()
    for _v in variables:
        min_value = df[_v].min()
        max_value = df[_v].max()
        df[_v] = (df[_v] - min_value) / (max_value - min_value)    
    
    x_range = variables

    plots = []
    
    for i, (by, group) in enumerate(df.groupby(by=cat)):

        p = label_plot(by, orien='h')
        p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
        p.min_border_top = 0
        p.min_border_bottom = 0
        p.outline_line_width = 1
        p.outline_line_color = 'black'
        
        plots.append(p)
    
        p = figure(
            x_range = x_range,
            y_range = [-0.05, 1.05],
            frame_width = PLOT_STYLE.FRAME_WIDTH,
            frame_height = PLOT_STYLE.FRAME_HEIGHT,
            sizing_mode='stretch_width'
        )

        xs = np.array([0.5 + i for i in range(len(variables))] * len(group)).reshape(len(group), len(variables))
        ys = group[variables].to_numpy()
        data = {
            'xs': xs.tolist(),
            'ys': ys.tolist()
        }

        p.multi_line(xs='xs', ys='ys', line_color='black', line_alpha=0.5, source=data)
        p.yaxis.ticker = [0, 0.5, 1]
        p.yaxis.major_label_overrides = {
            0: 'lowest',
            0.5: 'middle',
            1: 'highest'
        }

        # p.xaxis.axis_label = 'Fatty Acid'
        p.xaxis.axis_label_text_font_size = '12pt'
        p.axis.major_label_text_font_size = '10pt'
        p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
        p.min_border_right = 0
        p.min_border_top = PLOT_STYLE.PLOT_PADDING
        p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        p.outline_line_width = 1
        p.outline_line_color = 'black'

        plots.append(p)
                
    for _p in plots[1:-2:2]:
        _p.xaxis.visible = False
    
    grid = gridplot(plots, ncols=1, toolbar_location=None)
    return grid

source_code = '''
import pandas as pd
import numpy as np
import bokeh
from bokeh.plotting import figure, gridplot
from bokeh.io import output_notebook, show
from bokeh.palettes import brewer, Category20, Oranges, Reds, Purples
from bokeh.layouts import layout, grid, column, row
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import factor_cmap, linear_cmap, LinearColorMapper

output_notebook()

class PLOT_TYPE():
    BAR          =  1
    BOX          =  2
    HIST         =  3
    HIST_BY_CATE =  4
    SCATTER      =  5
    CORR         =  6
    NONE         =  0

class PLOT_STYLE():
    FRAME_WIDTH = 132
    FRAME_HEIGHT = 132
    MIN_BORDER_LEFT = 50
    PLOT_PADDING = 5
    TOOLS = None
    
def get_plot_type(df, row_variable, col_variable, row_idx, col_idx):
    dtype_row = df[row_variable].describe().dtype.name
    dtype_col = df[col_variable].describe().dtype.name
    
    if row_variable == col_variable:
        if dtype_row in ['object']:
            return PLOT_TYPE.BAR
        if dtype_row in ['float64']:
            return PLOT_TYPE.HIST
    else:
        if dtype_row in ['object'] and dtype_col in ['float64']:
            return PLOT_TYPE.BOX
        if dtype_row in ['float64'] and dtype_col in ['object']:
            return PLOT_TYPE.HIST_BY_CATE
        if dtype_row in ['float64'] and dtype_col in ['float64']:
            if row_idx > col_idx:
                return PLOT_TYPE.SCATTER
            else:
                return PLOT_TYPE.CORR
        
    return PLOT_TYPE.NONE

def vbar_plot(df, cat, colors):
    df_count = df.groupby(by=cat).agg({cat: 'count'})
    data = {
        'x': df_count.index.tolist(),
        'top': df_count[cat], 
        'colors': colors[len(df_count)]
    }
    p = figure(
        x_range      = data['x'],
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
        # x_axis_location='above'
    )
    p.vbar(x='x', top='top', width=0.9, fill_color='colors', line_width=0, source=data)
    p.y_range.start = 0
    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    return p

def generate_box_plot_data(df, category, column):

    # find the quartiles and IQR for each category
    groups = df[[column, category]].groupby(by=category, sort=False)
    # generate some synthetic time series for six different categories
    cats = []

    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    # find the outliers for each category
    def outliers(group):
        cat = group.name
        cats.append(cat)
        return group[(group[column] > upper.loc[cat][column]) | (group[column] < lower.loc[cat][column])][column]
    out = groups.apply(outliers).dropna()
    # prepare outlier data for plotting, we need coordinates for every outlier.
    out_index = []
    out_category = []
    out_value = []
    if not out.empty:
        for keys in out.index:
            out_index.append(keys[1])
            out_category.append(keys[0])
            out_value.append(out.loc[keys[0]].loc[keys[1]])

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)

    upper[column] = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,column]),upper[column])]
    lower[column] = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,column]),lower[column])]

    data_box_plot = {
        'category': cats,
        'qmin': qmin[column].tolist(),
        'q1' : q1[column].tolist(),
        'q2' : q2[column].tolist(),
        'q3' : q3[column].tolist(),
        'iqr': iqr[column].tolist(),
        'qmax': qmax[column].tolist(),
        'upper': upper[column].tolist(),
        'lower': lower[column].tolist()
    }

    data_box_plot_outlier = {
        'index': out_index,
        'category' : out_category,
        'value': out_value
    }
    
    return data_box_plot, data_box_plot_outlier

def box_plot(df, cat, cont, colors):
    
    categories = [by for by, group in df.groupby(by=cat)]
    colors_cat = colors[len(categories)]
    box, out = generate_box_plot_data(df, cat, cont)
    p = figure(
        y_range      = categories[::-1],
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
    )
    
    box = ColumnDataSource(data=box)
    # stems
    p.segment(x0='upper', y0='category', x1='q3', y1='category', line_color='black', source=box)
    p.segment(x0='lower', y0='category', x1='q1', y1='category', line_color='black', source=box)

    # boxes
    p.hbar(y='category', height=0.7, right='q2', left='q3', fill_color='white', fill_alpha=0.5, 
           line_color=factor_cmap('category', colors_cat, categories), line_width=2, 
           source=box)
    p.hbar(y='category', height=0.7, right='q1', left='q2', fill_color='white', fill_alpha=0.3, 
           line_color=factor_cmap('category', colors_cat, categories), line_width=2, 
           source=box)

    # whiskers (almost-0 height rects simpler than segments)
    p.dash(x='lower', y='category', size=10, angle=np.pi/2, line_width=1.5, line_color='black', source=box)
    p.dash(x='upper', y='category', size=10, angle=np.pi/2, line_width=1.5, line_color='black', source=box)

    # outliers
    p.circle(x='value', y='category', size=4, color="#F38630", fill_alpha=1, source=out)

    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    return p

def hist_plot(df, cont):
    p = figure(
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
    )
    hist, edges = np.histogram(
        df[cont], 
        # bins = 11,
        # range=(0, df[cont].max())
    )
    data = {
        'hist': hist,
        'left': edges[:-1],
        'right': edges[1:]
    }
    p.quad(
        top='hist', bottom=0, left='left', right='right',
        fill_color='dimgray', line_width=0,
        source=data
    )
    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    
    return p

def scatter_plot(df, cat, cont_1, cont_2, colors):
    
    colors_cat = colors[len(df[cat].unique())]
    p = figure(
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
    )
    
    for i, (by, group) in enumerate(df.groupby(by=cat)):
        
        data = {
            'x': group[cont_1],
            'y': group[cont_2]
        } 
        p.scatter(x='x', y='y', color=colors_cat[i], alpha=0.5, source=data)
        
    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    
    return p
    
def corr_plot(df, cont_1, cont_2):

    corr = df[[cont_1, cont_2]].corr()
    
    p = figure(
        frame_width  = PLOT_STYLE.FRAME_WIDTH,
        frame_height = PLOT_STYLE.FRAME_HEIGHT
    )

    p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
    p.text(x=0, y=0, y_offset=-10, text=['Corr:'], text_baseline='middle', align='center')
    p.text(x=0, y=0, y_offset=10, text=['%.3f' % corr.loc[cont_1, cont_2]], text_baseline='middle', align='center')
    return p

def label_plot(text, orien='h'):
    
    p = None
    
    if orien == 'h':
        p = figure(
            frame_width  = PLOT_STYLE.FRAME_WIDTH,
            frame_height = PLOT_STYLE.FRAME_HEIGHT // 4,
            tools=''
        )
        p.text(x=0, y=0, text=[text], text_baseline='middle', align='center', text_color='white')
        p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
        p.min_border_bottom = 0
    
    if orien == 'v':
        p = figure(
            frame_width  = PLOT_STYLE.FRAME_WIDTH // 4,
            frame_height = PLOT_STYLE.FRAME_HEIGHT,
            tools=''
        )
        p.text(x=0, y=0, text=[text], text_baseline='middle', align='center', text_color='white', angle=3*np.pi/2)
        p.min_border_left = 0
    
    p.background_fill_color = 'deepskyblue'
    p.grid.visible = False
    p.axis.visible = False
    p.toolbar.logo = None
    return p

def legend_plot(categories, variables, colors):
    
    colors_cat = colors[len(categories)][::-1]
    size = 10
    times = 20
    padding = 2
    frame_height = round(PLOT_STYLE.FRAME_HEIGHT * len(variables) + 
                         PLOT_STYLE.FRAME_HEIGHT // 4 + 
                         len(variables) * PLOT_STYLE.PLOT_PADDING * 2)
    p = figure(
        tools         = '', 
        frame_width   = size * times,
        frame_height  = frame_height,
        x_range       = [0, size * times],
        y_range       = [-(frame_height - PLOT_STYLE.FRAME_HEIGHT // 4 - PLOT_STYLE.PLOT_PADDING * 2) / 2, 
                          (frame_height - PLOT_STYLE.FRAME_HEIGHT // 4 - PLOT_STYLE.PLOT_PADDING * 2) / 2 + (PLOT_STYLE.FRAME_HEIGHT // 4 + PLOT_STYLE.PLOT_PADDING * 2)]
    )
    
    step = 0
    if len(categories) % 2 == 0:
        step = -size * (len(categories) / 2 + 1)
    else:
        step = -size * (len(categories) / 2 + 1.5)
        
    for i, c in enumerate(categories[::-1]):
        p.square(x=size, y=step, size=size, color=colors_cat[i])
        p.text(x=size, y=step, text=[c], text_font_size={'value': '10pt'}, text_baseline='middle', align='left', x_offset=size)
        step += size * padding

    p.axis.visible = False
    p.grid.visible = False
    p.outline_line_width = 0
    p.min_border_left = 0
    p.x_range.start = 0
    p.toolbar.logo = None
    
    return p

def decorate_plot_axis(p, length, i, j):

    # 如果是第一行标签
    if i == -1:
        if j != 0:
            p.min_border_left = PLOT_STYLE.PLOT_PADDING
            p.min_border_right = PLOT_STYLE.PLOT_PADDING
            p.min_border_top = PLOT_STYLE.PLOT_PADDING
            p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        else:
            p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT    
            p.min_border_right = PLOT_STYLE.PLOT_PADDING
            p.min_border_top = PLOT_STYLE.PLOT_PADDING
            p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        return
    
    # 如果是最右面的标签列
    if j == -1:
        p.min_border_left = PLOT_STYLE.PLOT_PADDING
        p.min_border_right = PLOT_STYLE.PLOT_PADDING
        p.min_border_top = PLOT_STYLE.PLOT_PADDING
        p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        return
    
    # 不是最后一行也不是第一列
    if length - 1 != i and j != 0:
        if type(p) == bokeh.plotting.Figure:
            p.axis.visible = False
            p.min_border_left = PLOT_STYLE.PLOT_PADDING
            p.min_border_right = PLOT_STYLE.PLOT_PADDING
            p.min_border_top = PLOT_STYLE.PLOT_PADDING
            p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        if type(p) == bokeh.models.layouts.GridBox:
            for p,_,_ in p.children:
                p.axis.visible = False
                p.min_border_left = PLOT_STYLE.PLOT_PADDING
                p.min_border_right = PLOT_STYLE.PLOT_PADDING
                p.min_border_top = PLOT_STYLE.PLOT_PADDING
                p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        return
    
    # 如果是第一列
    if j == 0:
        # 如果不是最后一行
        if i != length - 1:
            if type(p) == bokeh.plotting.Figure:
                p.xaxis.visible = False
                p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
                p.min_border_right = PLOT_STYLE.PLOT_PADDING
                p.min_border_top = PLOT_STYLE.PLOT_PADDING
                p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
            if type(p) == bokeh.models.layouts.GridBox:
                p.children[-1][0].min_border_right = PLOT_STYLE.PLOT_PADDING
                for p,_,_ in p.children:
                    p.min_border_top = PLOT_STYLE.PLOT_PADDING
                    p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
                    p.xaxis.visible = False
        # 如果是最后一行，也就是左下角的绘图
        else:
            if type(p) == bokeh.plotting.Figure:
                p.xaxis.visible = False
                p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
                p.min_border_right = PLOT_STYLE.PLOT_PADDING
                p.min_border_top = PLOT_STYLE.PLOT_PADDING
                p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
            if type(p) == bokeh.models.layouts.GridBox:
                p.children[-1][0].min_border_right = PLOT_STYLE.PLOT_PADDING
                for p,_,_ in p.children:
                    p.min_border_top = PLOT_STYLE.PLOT_PADDING
                    p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
                    p.xaxis.major_label_orientation = np.pi/2
                    p.xaxis.ticker.desired_num_ticks = 3
    # 如果不是第一列，那么这里就是最后一行
    else:
        if type(p) == bokeh.plotting.Figure:
            p.yaxis.visible = False
            p.min_border_left = PLOT_STYLE.PLOT_PADDING
            p.min_border_right = PLOT_STYLE.PLOT_PADDING
            p.min_border_top = PLOT_STYLE.PLOT_PADDING
            p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        if type(p) == bokeh.models.layouts.GridBox:
            p.children[-1][0].min_border_right = PLOT_STYLE.PLOT_PADDING
            p.children[0][0].min_border_left = PLOT_STYLE.PLOT_PADDING
            for p,_,_ in p.children:
                p.min_border_top = PLOT_STYLE.PLOT_PADDING
                p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
                p.xaxis.major_label_orientation = np.pi/2
                p.xaxis.ticker.desired_num_ticks = 3

def hist_by_cate_plot(df, cat, cont, colors):
    plots = []
    cat_length = len(df[cat].unique())
    cat_colors = colors[cat_length]
    
    x_max = 0
    for i, (by, group) in enumerate(df.groupby(by=cat)):
        p = figure(
            frame_width  = (PLOT_STYLE.FRAME_WIDTH - (cat_length - 1) * 2 * 2) // cat_length,
            frame_height = PLOT_STYLE.FRAME_HEIGHT,
            tools='save'
        )
        hist, edges = np.histogram(
            group[cont], 
            range=(0, df[cont].max())
        )
        hist_max = max(hist)
        if x_max < hist_max: x_max = hist_max
        data = {
            'hist': hist,
            'bottom': edges[:-1],
            'top': edges[1:]
        }
        p.quad(
            top='top', bottom='bottom', left=0, right='hist',
            fill_color=cat_colors[i], line_width=0,
            source=data
        )
        plots.append(p)

    for i, p in enumerate(plots):
        if i == 0: 
            p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
            p.min_border_right = 2
        else:
            p.min_border_left  = 2
            p.min_border_right = 2
            p.yaxis.visible = False

        p.xaxis.major_label_orientation = np.pi/2
        p.xaxis.ticker.desired_num_ticks = 3
        p.xaxis.ticker.num_minor_ticks = 0
        p.x_range.start = -x_max * 0.05
        p.x_range.end = x_max * 1.05
    
    grid = gridplot(plots, ncols=cat_length, toolbar_location=None)
    return grid

def pair_plot(df, variables):
    plots = np.full(
        [len(variables)+1, len(variables)+1], 
        None
    )
    # 行数据类型
    for i, _v_row in enumerate(variables):
        # 列数据类型
        for j, _v_col in enumerate(variables):
            plot_type = get_plot_type(df, _v_row, _v_col, i, j)
            p = None
            if plot_type == PLOT_TYPE.BAR:
                p = vbar_plot(df, _v_row, brewer['Set1'])
            if plot_type == PLOT_TYPE.BOX:
                p = box_plot(df, _v_row, _v_col, brewer['Set1'])
            if plot_type == PLOT_TYPE.HIST:
                p = hist_plot(df, _v_row)
            if plot_type == PLOT_TYPE.HIST_BY_CATE:
                p = hist_by_cate_plot(df, _v_col, _v_row, brewer['Set1'])
            if plot_type == PLOT_TYPE.SCATTER:
                p = scatter_plot(df, variables[0], _v_col, _v_row, brewer['Set1'])
            if plot_type == PLOT_TYPE.CORR:
                p = corr_plot(df, _v_col, _v_row)
            
            decorate_plot_axis(p, len(variables), i, j)
            plots[i+1, j] = p
    
    for i, v in enumerate(variables):
        p = label_plot(v, orien='h')
        decorate_plot_axis(p, len(variables), -1, i)
        plots[0, i] = p

    for i, v in enumerate(variables):
        p = label_plot(v, orien='v')
        decorate_plot_axis(p, len(variables), i, -1)
        plots[i+1, len(variables)] = p

    plots = plots.reshape(-1).tolist()

    grid = gridplot(plots, ncols=len(variables)+1, toolbar_location=None)
    p = legend_plot([by for by, group in df.groupby(by=variables[0])], variables, brewer['Set1'])
    
    return row(grid, p, sizing_mode='stretch_height')

def heatmap_plot(df, row_variables, col_variables, full=False):
    
    corr_array = np.full(
        (len(row_variables), len(row_variables)), 
        .0
    )
    for i in range(len(row_variables)):
        for j in range(len(col_variables)):
            corr = df[[row_variables[i], col_variables[j]]].corr()
            corr_array[i, j] = corr.iloc[0, 1]
            
    p = figure(
        x_range = col_variables,
        y_range = row_variables,
        frame_width = 400, frame_height = 400
    )

    row_num = corr_array.shape[0]
    col_num = corr_array.shape[1]
    
    if full == False:
        data = {
            'x': [_r + 0.5 for _c in range(col_num) for _r in range(_c + 1)],
            'y': [_c + 0.5 for _c in range(col_num) for _r in range(_c + 1)],
            'corr': [corr_array.reshape(-1)[_c * col_num + _r] for _c in range(col_num) for _r in range(_c + 1)]
        }
    else:
        data = {
            'x': [_c + 0.5 for _c in range(col_num)] * row_num,
            'y': [_p // col_num + 0.5 for _p in range(col_num * row_num)],
            'corr': corr_array.reshape(-1).tolist()
        }

    data['corr_text'] = ['%.2f' % _corr for _corr in data['corr']]
    
    palette = Purples[256] + Oranges[256][::-1]
    low = min(data['corr'])
    high = max(data['corr'])
    palette = palette[int((low - (-1)) / 2 * 512): int((high - (-1)) / 2 * 512)]
    alpha = 1.
    
    p.rect(
        x='x', y='y',
        width=1, height=1, 
        line_color='dimgrey', line_width=0.5, line_alpha=0.5,
        fill_color=linear_cmap('corr', palette, low=low, high=high),
        fill_alpha=alpha,
        source=data
    )

    p.text(
        x='x', y='y',
        text='corr_text', text_font_size={'value': '12px'},
        text_baseline='middle', align='center', 
        source=data
    )
    
    color_bar = ColorBar(
        color_mapper=LinearColorMapper(low=low, high=high, palette=palette),
        major_label_text_font_style='bold',
        major_tick_line_color='black',
        label_standoff=6, 
        border_line_color=None,
        location=(0,0)
    )
    p.add_layout(color_bar, 'right')

    p.x_range.range_padding = 0.03
    p.y_range.range_padding = 0.03
    p.xaxis.major_label_orientation = .8
    p.axis.axis_label_text_font_size = '14pt'
    p.axis.major_label_text_font_size = '10pt'
    
    return p

def parallel_plot(df, cat, variables):

    df = df.copy()
    for _v in variables:
        min_value = df[_v].min()
        max_value = df[_v].max()
        df[_v] = (df[_v] - min_value) / (max_value - min_value)    
    
    x_range = variables

    plots = []
    
    for i, (by, group) in enumerate(df.groupby(by=cat)):

        p = label_plot(by, orien='h')
        p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
        p.min_border_top = 0
        p.min_border_bottom = 0
        p.outline_line_width = 1
        p.outline_line_color = 'black'
        
        plots.append(p)
    
        p = figure(
            x_range = x_range,
            y_range = [-0.05, 1.05],
            frame_width = PLOT_STYLE.FRAME_WIDTH,
            frame_height = PLOT_STYLE.FRAME_HEIGHT,
            sizing_mode='stretch_width'
        )

        xs = np.array([0.5 + i for i in range(len(variables))] * len(group)).reshape(len(group), len(variables))
        ys = group[variables].to_numpy()
        data = {
            'xs': xs.tolist(),
            'ys': ys.tolist()
        }

        p.multi_line(xs='xs', ys='ys', line_color='black', line_alpha=0.5, source=data)
        p.yaxis.ticker = [0, 0.5, 1]
        p.yaxis.major_label_overrides = {
            0: 'lowest',
            0.5: 'middle',
            1: 'highest'
        }

        # p.xaxis.axis_label = 'Fatty Acid'
        p.xaxis.axis_label_text_font_size = '12pt'
        p.axis.major_label_text_font_size = '10pt'
        p.min_border_left = PLOT_STYLE.MIN_BORDER_LEFT
        p.min_border_right = 0
        p.min_border_top = PLOT_STYLE.PLOT_PADDING
        p.min_border_bottom = PLOT_STYLE.PLOT_PADDING
        p.outline_line_width = 1
        p.outline_line_color = 'black'

        plots.append(p)
                
    for _p in plots[1:-2:2]:
        _p.xaxis.visible = False
    
    grid = gridplot(plots, ncols=1, toolbar_location=None)
    return grid
'''