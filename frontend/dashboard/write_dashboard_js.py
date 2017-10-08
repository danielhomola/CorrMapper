import os
import pandas as pd
import numpy as np


def write_dashboard_js(dashboard_folder, autocorr, meta, min_med_max, pc_var, chart_dims, num_comp):
    """
    This epic sized function writes the custom JS that drives the dashboard.
    :param dashboard_folder: abs path to the folder where we save dashboard.js
    :param autocorr: whether we have two datasets or just one
    :param meta: open metadata pandas dataframe
    :param min_med_max: dictionary with min, median, max values of continuous
                        and PCA scores variables (ignoring the missing values)
    :param pc_var: principal components of PCA
    :param chart_dims: size of the charts, calculated by layout()
    :param num_comp: number of components to show in metadata explorer
    :return: dict of charts
    """

    f = open(os.path.join(dashboard_folder, 'dashboard.js'), 'w')

    # variables for proper indentation and legibility
    ind = '    '
    n1 = '\n' + ind
    n2 = '\n' + ind * 2
    n3 = '\n' + ind * 3
    n4 = '\n' + ind * 4

    # number of bins to use for continuous variables
    histo_bins = 15
    # if a categorical variables has longer string than this we use row-chart
    row_limit = 15
    # this dict will hold all widths and heights to do window.resize at the end
    chart_sizes = {}

    # get two level column variables from the clean metadata
    metacols = meta.columns.get_level_values(0)
    metatype = meta.columns.get_level_values(1)
    metaheader = pd.Series(metatype.values, index=metacols)

    # -------------------------------------------------------------------------
    # DEFINE VARIABLES
    # -------------------------------------------------------------------------

    f.write("d3.json(data_loc, function (error, data) {")
    f.write("%svar metadata = data;" % n1)
    f.write("%s_.each(metadata, function(d) {" % n1)

    f.write('\n%s// ===================================================\n' % n1)
    # CorrMapper ID
    f.write("%sd.CorrMapperID = +d.CorrMapperID;" % n2)

    # continuous variables
    if np.any(metatype == 'con'):
        for col in metacols[metatype == 'con']:
            f.write("%sd.%s = +d.%s;" % (n2, col, col))

    # Date
    if np.any(metatype == 'date'):
        f.write("%sd.Year = +d.Year;" % n2)
        f.write("%sd.Date = d3.time.month(new Date(d.Date));" % n2)

    # PCA coordinates
    f.write("%sd.dataset1_pc1 = +d.dataset1_pc1;" % n2)
    f.write("%sd.dataset1_pc2 = +d.dataset1_pc2;" % n2)
    if not autocorr:
        f.write("%sd.dataset2_pc1 = +d.dataset2_pc1;" % n2)
        f.write("%sd.dataset2_pc2 = +d.dataset2_pc2;" % n2)

    # end dashboard.json read
    f.write("%s});" % n1)

    # -------------------------------------------------------------------------
    # DEFINE CROSSFILTER AND CROSSFILTER DIMENSIONS
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================\n' % n1)
    # setup crossfilter
    f.write("%svar ndx = crossfilter(metadata);" % n1)

    # create dimensions for non-PC cols

    PC_cols = list(np.array([('dataset1_pc' + str(i + 1), 'dataset2_pc' + str(i + 1)) for i in range(num_comp)]).ravel())
    cols_without_PC = set(metacols) - set(PC_cols) - set(['ID', 'CorrMapperID'])
    for col in cols_without_PC:
        if metaheader[col] != 'con':
            f.write("%svar %sDim  = ndx.dimension(dc.pluck('%s'));" % (n1, col, col))
        else:
            # create custom binning for continuous variables to get nice  barcharts
            f.write("%svar %sRange = [%s, %s];"
                        % (n1, col, str(min_med_max[col]['min']),
                           str(min_med_max[col]['max'])))
            f.write("%svar %sBinWidth = (%sRange[1]-%sRange[0])/%d;"
                    %(n1, col, col, col, histo_bins))
            f.write("%svar %sDim = ndx.dimension(function(d) {"
                    "%svar %sThresholded = d.%s;"
                    "%sreturn %sBinWidth * Math.floor(%sThresholded / %sBinWidth);%s});"
                    %(n1, col,
                      n2, col, col,
                      n2, col, col, col, n1
                      ))

    # dimension for PC cols
    f.write("%svar dataset1Dim = ndx.dimension(function (d) "
            "{return [d.dataset1_pc1, d.dataset1_pc2];});" % n1)
    if not autocorr:
        f.write("%svar dataset2Dim = ndx.dimension(function (d) "
                "{return [d.dataset2_pc1, d.dataset2_pc2];});" % n1)

    # dimension for all counter
    f.write("%svar allDim = ndx.dimension(function(d) {return d;});" % n1)

    #  ------------------------------------------------------------------------
    # CUSTOM REDUCE FUNCTION FOR THE COLOURING OF THE PCA SCATTER PLOTS
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================\n' % n1)
    # rAdd
    f.write("%sfunction rAdd(p, v) {%s++p.count;" % (n1, n2))
    for col in list(cols_without_PC) + ['CorrMapperID']:
        f.write("%sp.color['%s'] = v.%s;" % (n2, col, col))
    f.write("%sreturn p;%s}" % (n2, n1))

    # rRemove
    f.write("%sfunction rRemove(p, v) {%s--p.count;" % (n1, n2))
    for col in list(cols_without_PC) + ['CorrMapperID']:
        f.write("%sp.color['%s'] = v.%s;" % (n2, col, col))
    f.write("%sreturn p;%s}" % (n2, n1))

    # rInit
    f.write("%sfunction rInit() {%sreturn {count: 0, color:{}};%s}" % (n1, n2, n1))

    # -------------------------------------------------------------------------
    # DEFINE CROSSFILTER GROUPS
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================\n' % n1)
    # Date
    if np.any(metatype == 'date'):
        f.write("%svar countPerDate = DateDim.group().reduceSum(function (d) "
            "{%sreturn 1;%s});" % (n1, n2, n1))

    # create groups for non-PC cols
    for col in cols_without_PC:
        if col != 'Date':
            f.write("%svar countPer%s = %sDim.group()"
                    ".reduceCount();" % (n1, col, col))

    # group for PC cols
    f.write("%svar countPerDataset1 = dataset1Dim.group()"
            ".reduce(rAdd,rRemove,rInit);" % n1)
    if not autocorr:
        f.write("%svar countPerDataset2 = dataset2Dim.group()"
                ".reduce(rAdd,rRemove,rInit);" % n1)

    # group for all counter
    f.write("%svar all = ndx.groupAll();" % n1)

    # -------------------------------------------------------------------------
    # DEFINE CHART TYPES
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================\n' % n1)
    pie_charts = []
    row_charts = []
    bar_charts = []
    scatter_charts = []

    # Date
    if np.any(metatype == 'date'):
        f.write("%svar YearChart = dc.pieChart('#chart-Year');" % n1)
        f.write("%svar MonthChart = dc.pieChart('#chart-Month');" % n1)
        f.write("%svar DayChart = dc.pieChart('#chart-Day');" % n1)
        f.write("%svar DateChart = dc.barChart('#chart-Date');" % n1)
        pie_charts += ['Year', 'Month', 'Day']
        bar_charts.append('Date')

    # Continuous bar charts
    if np.any(metatype == 'con'):
        for col in metacols[metatype == 'con']:
            f.write("%svar %sChart  = dc.barChart('#chart-%s');" % (n1, col, col))
            bar_charts.append(col)
    # Pie and row charts
    if np.any(metatype == 'cat'):
        for col in metacols[metatype == 'cat']:
            # if we have very long names in categorical variable use row chart
            unique_vals = map(len, map(str, pd.unique(pd.Series(meta.loc[:, col].values.ravel()))))
            if np.array(unique_vals).max() > row_limit:
                f.write("%svar %sChart = dc.rowChart('#chart-%s');" % (n1, col, col))
                row_charts.append(col)
            else:
                f.write("%svar %sChart = dc.pieChart('#chart-%s');" % (n1, col, col))
                pie_charts.append(col)
        patient_col = ''
    if np.any(metatype == 'patient'):
        patient_col = metacols[(metatype == 'patient')].values[0]
        f.write("%svar %sChart = dc.pieChart('#chart-%s');"
                % (n1, patient_col, patient_col))
        pie_charts.append(patient_col)

    # PCA scatter plot(s)
    f.write("%svar dataset1Chart = dc.scatterPlot('#chart-dataset1');" % n1)
    scatter_charts.append('dataset1')
    if not autocorr:
        f.write("%svar dataset2Chart = dc.scatterPlot('#chart-dataset2');" % n1)
        scatter_charts.append('dataset2')

    # datatable, data count
    f.write("%svar dataCount = dc.dataCount('#data-count');" % n1)
    f.write("%svar dataTable = dc.dataTable('#data-table');" % n1)

    # -------------------------------------------------------------------------
    # WRITE PIE CHARTS
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================' % n1)
    # Pie chart
    for chart in pie_charts:
        if chart in ['Year', 'Month', 'Day']:
            # date related pie charts need different color scheme
            f.write("\n%svar %sColor = d3.scale.category20();" % (n1, chart))
        elif chart == patient_col:
            f.write("\n%svar %sColor = d3.scale.category20();" % (n1, chart))
        else:
            f.write("\n%svar %sColor = d3.scale.category10();" % (n1, chart))

        f.write("%s%sChart"
                "%s.width(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)"
                "%s.height(%d)"
                "%s.dimension(%sDim)"
                "%s.group(countPer%s)"
                "%s.colors(function (d) {return %sColor(d);})"
                "%s.innerRadius(%d)"
                % (n1, chart,
                   n2, chart,
                   n2, chart_dims['pie_chart_width'],
                   n2, chart,
                   n2, chart,
                   n2, chart,
                   n2, chart_dims['pie_chart_radius']))
        # months and days are ordered in the pie chart
        if chart == 'Month':
            f.write("%s.ordering(function (d) {"
                    "%svar order = {"
                    "%s'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, "
                    "%s'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, "
                    "%s'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12"
                    "%s};"
                    "%sreturn order[d.key];"
                    "%s})" % (n2, n3, n4, n4, n4, n3, n3, n2))
        if chart == 'Day':
            f.write("%s.ordering(function (d) {"
                    "%svar order = {"
                    "%s'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,"
                    "%s'Fri': 4, 'Sat': 5, 'Sun': 6"
                    "%s};"
                    "%sreturn order[d.key];"
                    "%s})"
                    % (n2, n3, n4, n4, n3, n3, n2))
        f.write(";")

        chart_sizes[chart] = {
            "width": ".width(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)" % chart,
            "height": ".height(%d)" % chart_dims['pie_chart_width']
        }

    # -------------------------------------------------------------------------
    # WRITE ROW CHARTS
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================' % n1)
    for chart in row_charts:
        f.write("\n%svar %sColor = d3.scale.category10();" % (n1, chart))
        f.write("%s%sChart"
                "%s.width(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)"
                "%s.height(%d)"
                "%s.dimension(%sDim)"
                "%s.group(countPer%s)"
                "%s.colors(function (d) {return %sColor(d);});"
                % (n1, chart,
                   n2, chart,
                   n2, chart_dims['pie_chart_width'],
                   n2, chart,
                   n2, chart,
                   n2, chart))

        chart_sizes[chart] = {
            "width": ".width(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)" % chart,
            "height": ".height(%d)" % chart_dims['pie_chart_width']
        }

    # -------------------------------------------------------------------------
    # WRITE BAR CHARTS
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================' % n1)
    # write min, median, max
    for chart in bar_charts:
        # for charts but not for date
        f.write('\n')
        mmm = ['min', 'median', 'max']
        for mi, m in enumerate(mmm):
            if chart != 'Date':
                f.write("%svar %s%s = %s;"
                        % (n1, m, chart, str(min_med_max[chart][m])))
            else:
                # min, median, max for date
                f.write("%svar %s%s = new Date('%s');"
                        % (n1, m, chart, str(min_med_max[chart][m])))

        # colour for continuous vars not date
        f.write("%svar %sColor = d3.scale.linear()"
                "%s.domain([min%s, d3.mean([min%s, median%s]), median%s, d3.mean([max%s, median%s]), max%s, 2*max%s])"
                "%s.range(['#144c73', '#1D70AB', '#fff','#921113', '#d91a1d', '#EAE1F0'])"
                "%s.interpolate(d3.interpolateHsl);"
                % (n1, chart,
                   n2, chart, chart, chart, chart, chart, chart, chart, chart,
                   n2,
                   n2))

        f.write("%s%sChart"
                "%s.width(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)"
                "%s.height(%d)"
                "%s.dimension(%sDim)"
                "%s.group(countPer%s)"
                "%s.elasticY(true)"
                "%s.barPadding(1)"
                "%s.renderHorizontalGridLines(true)"
                "%s.margins({top: 10, right: 10, bottom: 20, left: 30})"
                % (n1, chart,
                   n2, chart,
                   n2, chart_dims['pie_chart_width'],
                   n2, chart,
                   n2, chart,
                   n2, n2, n2, n2))

        if chart != 'Date':
            f.write("%s.x(d3.scale.linear().domain([min%s, max%s]))"
                    "%s.xUnits(dc.units.fp.precision(%sBinWidth))"
                    "%s.round(function(d) {%sreturn %sBinWidth * Math.floor(d / %sBinWidth)%s});"
                    % (n2, chart, chart, n2, chart, n2, n3, chart, chart, n2))
        else:
            f.write("%s.x(d3.time.scale().domain([new Date(minDate), new Date(maxDate)]))"
                    "%s.round(d3.time.month.round)"
                    "%s.alwaysUseRounding(true)"
                    "%s.xUnits(d3.time.months);"
                    % (n2, n2, n2, n2))

        chart_sizes[chart] = {
            "width": ".width(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)" % chart,
            "height": ".height(%d)" % chart_dims['pie_chart_width']
        }

    # -------------------------------------------------------------------------
    # PCA PLOT(S)
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================' % n1)
    pcs_list = ['pc' + str(pc + 1) for pc in range(num_comp)]
    minmax = ['min', 'max']
    for i, dataset in enumerate(scatter_charts):
        f.write("\n%svar %s_pcs = {};" % (n1, dataset))
        for pci, pc in enumerate(pcs_list):
            for m in minmax:
                f.write("%s%s_pcs['%s_%s'] = %s*1.1;"
                        % (n1, dataset, m, pc, str(min_med_max[dataset][pc + '_' + m])))
            f.write("%s%s_pcs['var_%s'] = 'PC%d %s%s'"
                    % (n1, dataset, pc, pci + 1,
                       "{:0.2f}".format(pc_var[dataset]['pc' + str(pci + 1)]), '%'))

        f.write("%s%sChart"
                "%s.width(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)"
                "%s.height(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)"
                "%s.dimension(%sDim)"
                "%s.group(countPer%s)"
                "%s.margins({top: 20, right: 10, bottom: 40, left: 40})"
                "%s.x(d3.scale.linear().domain([%s_pcs['min_pc1'], %s_pcs['max_pc1']]))"
                "%s.y(d3.scale.linear().domain([%s_pcs['min_pc2'], %s_pcs['max_pc2']]))"
                "%s.xAxisLabel(%s_pcs['var_pc1'])"
                "%s.yAxisLabel(%s_pcs['var_pc2'])"
                "%s.clipPadding(10)"
                "%s.symbolSize(10)"
                "%s.renderHorizontalGridLines(true)"
                "%s.renderVerticalGridLines(true)"
                "%s.colorAccessor(function(d) {return d.value.color[d3.select('#%s_colour').property('value')];})"
                "%s.colors(function (d) {return colourSwitcher%d(d);})"
                "%s.existenceAccessor(function(d) { return d.value.count; });"
                % (n1, dataset,
                   n2, dataset,
                   n2, dataset,
                   n2, 'dataset' + str(i + 1),
                   n2, 'Dataset' + str(i + 1),
                   n2,
                   n2, dataset, dataset,
                   n2, dataset, dataset,
                   n2, dataset,
                   n2, dataset,
                   n2,
                   n2,
                   n2,
                   n2,
                   n2, dataset,
                   n2, i + 1,
                   n2))

        chart_sizes[dataset] = {
            "width": ".width(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)" % dataset,
            "height": ".height(d3.select('#chart-%s').node().parentNode.getBoundingClientRect().width * 0.9)" % dataset
        }

    # -------------------------------------------------------------------------
    # DATA COUNT & DATA TABLE
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================\n' % n1)
    # data counter
    f.write("%sdataCount%s.dimension(ndx)%s.group(all);" % (n1, n2, n2))

    # data table
    if np.any(metatype == 'date'):
        f.write("%svar dateFormat = d3.time.format('%s');" % (n1, '%Y-%m-%d'))
    f.write("\n%sdataTable%s.dimension(allDim)"
            "%s.group(function (d) { return '';})%s.size(%d)"
            %(n1, n2, n2, n2, meta.shape[0]))
    f.write("%s.columns([" % n2)

    # build data table columns
    datatable_cols = ['ID']
    if np.any(metatype == 'patient'):
        datatable_cols += list(metacols[metatype == 'patient'].values)
    datatable_cols += list(metacols[(metatype == 'con') + (metatype == 'cat')].values)
    if np.any(metatype == 'date'):
        datatable_cols += ['Date', 'Year', 'Month', 'Day']

    for col in datatable_cols:
        if col == "Date":
            f.write("%sfunction (d) { return dateFormat(d.%s); }," % (n3, col))
        else:
            f.write("%sfunction (d) { return d.%s; }," % (n3, col))
    f.write("%s])%s.sortBy(dc.pluck('ID'))%s.order(d3.ascending)"
            % (n2, n2, n2))
    f.write("%s.on('renderlet', function (table) "
            "{%stable.select('tr.dc-table-group').remove();%s});"
            % (n2, n3, n2))

    # -------------------------------------------------------------------------
    # COLOUR AND PCA SWITCHING OF THE PCA(s)
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================' % n1)
    for i, dataset in enumerate(scatter_charts):
        d = i + 1
        if d == 1:
            other_d = 2
        else:
            other_d = 1

        # COLOUR SWITCHING
        if autocorr:
            f.write("\n%sd3.select('#dataset%d_colour').on('change',function(){"
                "%sdataset%dChart.redraw();%s});" % (n1, d, n2, d, n1))
        else:
            f.write("\n%s$('#dataset%d_colour').change(function(){"
                    "%sif(document.getElementById('colour_lock').checked) {"
                    "%s$('#dataset%d_colour').val($(this).val());"
                    "%sdataset%dChart.redraw();"
                    "%s}"
                    "%sdataset%dChart.redraw();"

                    "%s});"
                    % (n1, d,
                       n2,
                       n3, other_d,
                       n3, other_d,
                       n2,
                       n2, d,
                       n1))

        f.write("%sfunction colourSwitcher%d(val){"
                    "%svar pca_colour = d3.select('#%s_colour').property('value');"
                    "%sswitch(pca_colour) {"
                    % (n1, i + 1, n2, dataset, n2))

        for col in cols_without_PC:
            f.write("%scase '%s':"
                    "%sreturn %sColor(val);"
                    % (n3, col, n4, col))
        f.write("%s}%s}" %(n2, n1))

        if d == 2:
            f.write("\n%s$('#colour_lock').change(function(){"
                    "%sif(document.getElementById('colour_lock').checked) {"
                    "%s$('#dataset2_colour').val($('#dataset1_colour').val());"
                    "%sdataset2Chart.redraw();"
                    "%s}"
                    "%s});"
                    % (n1,
                       n2,
                       n3,
                       n3,
                       n2,
                       n1))

        # PC SWITCHING
        f.write("\n%s$('#dataset%d_pcx').change(dataset%d_pc_xy);" % (n1, d, d))
        f.write("%s$('#dataset%d_pcy').change(dataset%d_pc_xy);" % (n1, d, d))

        f.write("%sfunction dataset%d_pc_xy (){"
                    "%sif(document.getElementById('pc_lock').checked) {"
                        "%sif ($('#dataset%d_pcx').val() != $('#dataset%d_pcx').val() ||"
                            "%s$('#dataset%d_pcy').val() != $('#dataset%d_pcy').val()) {"
                                "%s$('#dataset%d_pcx').val($('#dataset%d_pcx').val());"
                                "%s$('#dataset%d_pcy').val($('#dataset%d_pcy').val());"
                                "%sdataset%d_pc_xy();"
                        "%s}"
                    "%s}"
                    %(  n1, d,
                        n2,
                        n3, d, other_d,
                        n4, d, other_d,
                        n4, other_d, d,
                        n4, other_d, d,
                        n4, other_d,
                        n3,
                        n2))
        f.write(    "%svar x = $('#dataset%d_pcx').val();"
                    "%svar y = $('#dataset%d_pcy').val();"
                    "%svar filteredPCA = dataset%dDim.top(Infinity);"
                    "%svar new_pcx = filteredPCA.filter(function(obj){ return obj['dataset%d_pc' + x] <= dataset%d_pcs['max_pc' + x]});"
                    "%svar new_pcy = filteredPCA.filter(function(obj){ return obj['dataset%d_pc' + y] <= dataset%d_pcs['max_pc' + y]});"
                    "%snew_pcx = new_pcx.map(function(obj){return obj['dataset%d_pc' + x]});"
                    "%snew_pcy = new_pcy.map(function(obj){return obj['dataset%d_pc' + y]});"
                    "%svar resizeTimer;"
                    %(  n2, d,
                        n2, d,
                        n2, d,
                        n2, d, d,
                        n2, d, d,
                        n2, d,
                        n2, d,
                        n2))
        f.write(    "%sif(filteredPCA.length != %d && filteredPCA.length != 0 && new_pcx.length > 0 && new_pcy.length > 0){"
                        "%svar new_pcx_max = Math.max.apply(null, new_pcx);"
                        "%svar new_pcx_min = Math.min.apply(null, new_pcx);"
                        "%svar new_pcy_max = Math.max.apply(null, new_pcy);"
                        "%svar new_pcy_min = Math.min.apply(null, new_pcy);"
                        "%snew_pcx_max += Math.abs(new_pcx_max)*.1;"
                        "%snew_pcx_min -= Math.abs(new_pcx_min)*.1;"
                        "%snew_pcy_max += Math.abs(new_pcy_max)*.1;"
                        "%snew_pcy_min -= Math.abs(new_pcy_min)*.1;"
                        "%sdataset%dDim.dispose();"
                        "%sdataset%dDim = ndx.dimension(function (d) {return [d['dataset%d_pc' + x], d['dataset%d_pc' + y]];});"
                        "%scountPerDataset%d = dataset%dDim.group().reduce(rAdd,rRemove,rInit);"
                        "%sdataset%dChart"
                            "%s.dimension(dataset%dDim)"
                            "%s.group(countPerDataset%d)"
                            "%s.x(d3.scale.linear().domain([dataset%d_pcs['min_pc' + x], dataset%d_pcs['max_pc' + x]]))"
                            "%s.y(d3.scale.linear().domain([dataset%d_pcs['min_pc' + y], dataset%d_pcs['max_pc' + y]]))"
                            "%s.xAxisLabel(dataset%d_pcs['var_pc' + x])"
                            "%s.yAxisLabel(dataset%d_pcs['var_pc' + y])"
                            "%s.filter(null)"
                            "%s.filter(dc.filters.RangedTwoDimensionalFilter([[new_pcx_min, new_pcy_min], [new_pcx_max, new_pcy_max]]))"
                            "%s.redraw();"
                        "%sclearTimeout(resizeTimer);"
                        "%sresizeTimer = setTimeout(function() {"
                            "%sdc.redrawAll();"
                        "%s}, 100)"
                    %(  n2, meta.shape[0],
                        n3,
                        n3,
                        n3,
                        n3,
                        n3,
                        n3,
                        n3,
                        n3,
                        n3, d,
                        n3, d, d, d,
                        n3, d, d,
                        n3, d,
                        n4, d,
                        n4, d,
                        n4, d, d,
                        n4, d, d,
                        n4, d,
                        n4, d,
                        n4,
                        n4,
                        n4,
                        n3,
                        n3,
                        n4,
                        n3))
        f.write(    "%s} else{"
                        "%sdataset%dDim.dispose();"
                        "%sdataset%dDim = ndx.dimension(function (d) {return [d['dataset%d_pc' + x], d['dataset%d_pc' + y]];});"
                        "%scountPerDataset%d = dataset%dDim.group().reduce(rAdd,rRemove,rInit);"
                        "%sdataset%dChart.filterAll();"
                        "%sdc.redrawAll();"
                        "%sdataset%dChart"
                            "%s.dimension(dataset%dDim)"
                            "%s.group(countPerDataset%d)"
                            "%s.x(d3.scale.linear().domain([dataset%d_pcs['min_pc' + x], dataset%d_pcs['max_pc' + x]]))"
                            "%s.y(d3.scale.linear().domain([dataset%d_pcs['min_pc' + y], dataset%d_pcs['max_pc' + y]]))"
                            "%s.xAxisLabel(dataset%d_pcs['var_pc' + x])"
                            "%s.yAxisLabel(dataset%d_pcs['var_pc' + y])"
                            "%s.filterAll()"
                            "%s.redraw();"
                    "%s}"
                "%s}"
                    %(  n2,
                        n3, d,
                        n3, d, d, d,
                        n3, d, d,
                        n3, d,
                        n3,
                        n3, d,
                        n4, d,
                        n4, d,
                        n4, d, d,
                        n4, d, d,
                        n4, d,
                        n4, d,
                        n4,
                        n4,
                        n2,
                        n1))

        if d == 2:
            f.write("\n%s$('#pc_lock').change(function(){"
                    "%sif(document.getElementById('pc_lock').checked) {"
                    "%s$('#dataset2_pcx').val($('#dataset1_pcx').val());"
                    "%s$('#dataset2_pcy').val($('#dataset1_pcy').val());"
                    "%sdataset2_pc_xy();"
                    "%s}"
                    "%s});"
                    % (n1,
                       n2,
                       n3,
                       n3,
                       n4,
                       n2,
                       n1))

    # -------------------------------------------------------------------------
    # REGISTER RESET BUTTON HANDLERS AND RENDER ALL
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================\n' % n1)
    # write reset for 'all'
    f.write("%sd3.selectAll('a#%s').on('click', function () "
            "{%sdc.filterAll();%sdc.renderAll();%s});"
            % (n1, 'all', n2, n2, n1))
    # write all other
    for chart in pie_charts + row_charts + bar_charts + scatter_charts:
        f.write("%sd3.selectAll('a#%s').on('click', function () "
                "{%s%sChart.filterAll();%sdc.redrawAll();%s});"
                % (n1, chart, n2, chart, n2, n1))

    f.write("\n%sdc.renderAll();" % n1)

    # -------------------------------------------------------------------------
    # LAUNCH BOOSTRO TOUR
    # -------------------------------------------------------------------------

    f.write("%sif (dashboard_intro == 1) {"
            "%shelp();"
            "%s}"
            % (n1,
               n2,
               n1))

    # -------------------------------------------------------------------------
    # LET'S MAKE THE WHOLE THING RESPONSIVE
    # -------------------------------------------------------------------------

    f.write('\n%s// ===================================================\n' % n1)
    f.write("%svar resizeTimer;" % n1)
    f.write("%s$(window).on('resize', function(e) {" % n1)

    # we need to delay it a bit, so it runs at the END of resize event
    f.write("%sclearTimeout(resizeTimer);"
            "%sresizeTimer = setTimeout(function() {"
            % (n2, n2))

    for chart in chart_sizes:
        f.write("%s%sChart"
                "%s%s"
                "%s%s"
                % (n3, chart,
                   n4, chart_sizes[chart]['width'],
                   n4, chart_sizes[chart]['height']))

        if chart in bar_charts + scatter_charts:
            f.write("%s.rescale()%s.redraw();" % (n4, n4))
        else:
            f.write("%s.redraw();" % (n4))
    f.write("%s}, 100);%s});" % (n2,n1))

    f.write("\n});")
    f.close()

    # -------------------------------------------------------------------------
    # WRITE RETURN VALUES
    # -------------------------------------------------------------------------

    charts = {
        'cat_charts': list(set(pie_charts) - set(['Year', 'Month', 'Day'])),
        'con_charts': list(set(bar_charts) - set(['Date'])) + row_charts,
        'scatter_charts': scatter_charts
    }
    charts['all_charts'] = charts['cat_charts'] + charts['con_charts']

    dash_vars = {}
    dash_vars['autocorr'] = autocorr
    dash_vars['cols'] = datatable_cols

    if 'Date' in bar_charts:
        dash_vars['date'] = True
        charts['all_charts'] += ['Date']
    else:
        dash_vars['date'] = False
    return charts, dash_vars
