// ------------------------------------------------------------------------
// INTRO TO VIS GENOMIC
if (vis_genomic_intro == 1){
    $( document ).ready(help);
}

// ----------------------------------------------------------------------------
//
// NETWORK VISUALISATION
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// BASIC VARS AND INIT FUNCTION
var nw, ww, wh, diameter, offsetX, width, height;
var radius, innerRadius, cluster, svg, link, label, node;

var bundle = d3.layout.bundle();
var colorScale = d3.scale.linear()
    .domain([minCorr, 0, maxCorr])
    .range(["#0971B2", "#f2d9e5", "#CE1212"]);

// racalculates size and placement of network
function networkInit() {
// we have custom bootstrap that changes from md to lg at 1400 px width
    nw = $('#network').width();
    ww = $(window).width() - 100;
    wh = $(window).height() - 330;

    if (ww + 100 > 1400) {
        diameter = nw;
        offsetX = diameter / 2;
        width = diameter;
        height = diameter;
    } else {
        diameter = wh;
        offsetX = ww / 2;
        width = ww;
        height = diameter;
    }

    radius = diameter / 2;
    innerRadius = radius - 160;
    cluster = d3.layout.cluster()
        .size([360, innerRadius])
        .sort(null);

    svg = d3.select("#network").append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(" + offsetX + "," + radius *.97 + ")");

    link = svg.append("g").selectAll(".link");
    label = svg.append("g").selectAll(".label");
    node = svg.append("g").selectAll(".node");
}

// ----------------------------------------------------------------------------
// BUILD HIERARCHY FROM LISTS OF OBJECTS IN NETWORK.JS
// ----------------------------------------------------------------------------

function getGraph(){
    var nodes = [];
	var links = [];

    // we need a root
	var root = {
		name: "root",
		children: []
	};
	nodes.push(root);

    // add parents, i.e. chromosomes in our case
	for (var i = 0; i < parentsData.length; i ++) {
		var np = {
			name: parentsData[i].name,
            chr_num: parentsData[i].chr_num,
			parent: root,
			children: []
		};
		nodes.push(np);
		root.children.push(np);
	}

    // add children, i.e. chromosome bins
	for (var i = 0; i < nodesData.length; i ++) {
		var nd = {
			name: nodesData[i].name,
            chr_num: nodesData[i].chr_num,
            nodeSize: nodesData[i].nodeSize,
			parent: nodes[nodesData[i].chr_num + 1]
		};
		nodes.push(nd);
		nodes[nodesData[i].chr_num + 1].children.push(nd);
	}

    // we have to find the nodes in our list by its name
    function findNode(name, nodes){
        for (var i = 0; i < nodes.length; i ++) {
            if (nodes[i].name === name){
                return nodes[i];
            }
        }
    }

    // add edges
	for (var i = 0; i < edgesData.length; i ++) {
		links.push({
			source: findNode(edgesData[i].source, nodes),
			target: findNode(edgesData[i].target, nodes),
			corr: edgesData[i].edgeCorr,
			width: edgesData[i].edgeWidth,
            linkNum: edgesData[i].edgeNum
		});
	}

	return {
		nodes: nodes,
		links: links
	};
}

// ----------------------------------------------------------------------------
// DRAW GRAPH
// ----------------------------------------------------------------------------

var graph = getGraph();
networkInit();
drawNodes();
drawEdges(.85, -1, 1, 0, 5);

function drawNodes(){
    var nodes = cluster.nodes(graph.nodes[0]);

    label = label
        .data(nodes.filter(function (n) {
            return !n.children;
        }))
        .enter().append("text")
        .attr("class", "label")
        .attr("id", function (d) {
            return d.name;
        })
        .attr("dy", ".31em")
        .attr("transform", function (d) {
            return "rotate(" + (d.x - 90) + ")translate(" + (d.y + 20) + ",0)" + (d.x < 180 ? "" : "rotate(180)");
        })
        .style("cursor", "pointer")
        .style("text-anchor", function (d) {
            return d.x < 180 ? "start" : "end";
        })
        .text(function (d) {
            return d.name;
        })
        .on("click", networkClick)
        .on("mouseover", networkMouseOver)
        .on("mouseout", networkMouseOut);

    node = node
        .data(nodes.filter(function (n) {
            return !n.children;
        }))
        .enter()
        .append("circle")
        .attr("class", "node")
        .attr("dy", ".31em")
        .attr("transform", function (d) {
            return "rotate(" + (d.x - 90) + ")translate(" + (d.y + 10) + ",0)" + (d.x < 180 ? "" : "rotate(180)");
        })
        .attr("r", function (d) {
            return d.nodeSize;
        });
}

function drawEdges(tension, minCorr, maxCorr, minLink, maxLink) {
    var line = d3.svg.line.radial()
    .interpolate("bundle")
    .tension(tension)
    .radius(function (d) {
        return d.y;
    })
    .angle(function (d) {
        return d.x / 180 * Math.PI;
    });

    var links = bundle(graph.links);

    link = link
        .data(graph.links)
        .enter().append("path")
        .attr("class", "link")
        .attr("stroke-width", (function (d) {
            return d.width;
        }))
        .attr("stroke", (function (d) {
            return colorScale(d.corr);
        }))
        .attr("opacity", .65)
        .attr("d", function (d, i) {
            return line(links[i]);
        })
        .classed("linkFiltered", function(d){
            if (d.corr < minCorr || d.corr > maxCorr || d.linkNum < minLink || d.linkNum > maxLink){
                return true
            }
        });
}

// ----------------------------------------------------------------------------
// DEFINE MOUSE EVENTS AND HANDLERS
// ----------------------------------------------------------------------------

// special function so we can move the active (hovered) edges to the front
d3.selection.prototype.moveToFront = function () {
    return this.each(function () {
        this.parentNode.appendChild(this);
    });
};

var networkClicked = '';
function networkClick(d) {
    // reset tables
    resetTables();
    // redraw network with the click bin highlighted
    networkClicked = d.name;
    networkMouseOver(d.name);

    // update table titles
    $("#tab1").text(d.name);
    $("#tab2").text(d.name);

    // filter tables to the clicked bin
    table1.column(0).search(d.name).draw();
    table1.order([[5, "asc"], [6, "asc"], [7, "asc"]]).draw()
    table2.column(0).search(d.name).draw();
    table2.order([[5, "asc"], [6, "asc"], [7, "asc"]]).draw()

    // close previously opened bins
    if (openedTable1) {
        openedTable1.removeClass('info');
        table1.row(openedTable1).child.hide();
    }
    if (openedTable2) {
        openedTable2.removeClass('info');
        table2.row(openedTable2).child.hide();
    }
    openedTable1 = '';
    openedTable2 = '';
}

function networkMouseOver(d) {
    // to make sure this could be triggered from tables as well (they send string)
    if (typeof d === 'object') {
        d = d.name;
    }

    label.each(function (n) {
        n.target = n.source = false;
    });

    link
        .classed("linkNotOver", function (l) {
            if (l.target.name !== d && l.source.name !== d) {
                l.source.source = false;
                l.target.target = false;
                return true
            }
        })
        .classed("linkOver", function (l) {
            if (l.target.name === d) {
                l.source.source = true;
                return true;
            }
        })
        .classed("linkOver", function (l) {
            if (l.source.name === d) {
                l.target.target = true;
                return true
            }
        });
    d3.selectAll(".linkOver").moveToFront();

    label
        .classed("labelOver", function (l) {
            if (l.target || l.source) {
                return true;
            }
        })
        .classed("labelNotOver", function (l) {
            if (!l.target && !l.source && l.name !== d) {
                return true;
            }
        })
        .classed("labelSource", function (l) {
            if (l.name === d) {
                return true;
            }
        });
}

function networkMouseOut() {
    // reset
    if (networkClicked === ''){
        link.classed("linkNotOver", false);
        link.classed("linkOver", false);
        label.classed("labelNotOver", false);
        label.classed("labelOver", false);
        label.classed("labelSource", false);
    }else {

        // keep clicked links highlighted
        link
            .classed("linkNotOver", function (l) {
            if (l.target.name !== networkClicked && l.source.name !== networkClicked) {
                return true
            } else {
                l.source.source = true;
                l.target.target = true;
                return false
            }
        })
            .classed("linkOver", function (l) {
            if (l.target.name === networkClicked || l.source.name === networkClicked) {
                return true;
            } else {
                return false;
            }
        });

        label
            .classed("labelOver", function (l) {
            if (l.target || l.source) {
                return true;
            } else {
                return false;
            }
        })
            .classed("labelNotOver", function (l) {
            if (!l.target && !l.source && l.name !== networkClicked) {
                return true;
            } else {
                return false;
            }
        })
            .classed("labelSource", function (l) {
            if (l.name === networkClicked) {
                return true;
            } else {
                return false;
            }
        });
    }
}

// ------------------------------------------------------------------------
// RESPONSIVENESS
// ----------------------------------------------------------------------------

var resizeTimer;
$(window).on('resize', function(e) {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
        $('#network').empty();
        networkInit();
        drawNodes();
        // get values of sliders
        getSliderParams();
        drawEdges(tension, minCorr, maxCorr, minLink, maxLink);
    }, 200);
});

// ----------------------------------------------------------------------------
// SLIDER HANDLERS
// ----------------------------------------------------------------------------

// redraw edges if slider is moved
var tension, minCorr, maxCorr, minLink, maxLink;
function getSliderParams(){
    tension = $('#edgeBundlingSlider').data('slider').getValue()/100;
    minCorr = $('#corrFilterSlider').data('slider').getValue()[0];
    maxCorr = $('#corrFilterSlider').data('slider').getValue()[1];
    minLink = $('#numLinksSlider').data('slider').getValue()[0];
    maxLink = $('#numLinksSlider').data('slider').getValue()[1];
}

function sliderHandler(){
    // get values of sliders
    getSliderParams();
    // remove previous edges
    link.remove();
    resetTables();
    link = svg.append("g").selectAll(".link");
    // redraw edges
    drawEdges(tension, minCorr, maxCorr, minLink, maxLink);
}

// custom search function that will filter the tables by their corr values
$.fn.dataTable.ext.search.push(
    function( settings, data, dataIndex ) {
        var min = $('#corrFilterSlider').data('slider').getValue()[0];
        var max = $('#corrFilterSlider').data('slider').getValue()[1];
        var corr_str = data[2].split('__|__');
        var corr = corr_str.map(function(x){return parseFloat(x)});

        if ( ( isNaN( min ) && isNaN( max ) ) ||
             ( isNaN( min ) && corr.some(function(e){return e <= max}) ) ||
             ( corr.some(function(e){return e >= min}) && isNaN( max ) ) ||
             ( corr.some(function(e){return e >= min}) &&
               corr.some(function(e){return e<= max}) ) )
        {
            return true;
        }
        return false;
    }
);


$('#edgeBundlingSlider').slider().on('slide', sliderHandler);
$('#corrFilterSlider').slider().on('slide', sliderHandler);
$('#numLinksSlider').slider().on('slide', sliderHandler);

// ----------------------------------------------------------------------------
//
// TABLES
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// DEFINE TABLES

// populate table dynamically with cols we got from user
var colsBase = [
    {visible: false, "data": "CorrMapperBin"},
    {visible: false, "data": "CorrMapperCorrName"},
    {visible: false, "data": "CorrMapperRval"},
    {visible: false, "data": "CorrMapperPval"}];
var cols = colsBase.slice();
for (var i = 0; i < table1_cols.length; i++) {
    cols.push({title: table1_cols[i], "data": table1_cols[i], "type": "natural"})
}
var table1 = $('#table1').DataTable({
    ajax: table1_loc,
    columns: cols,
    // order by chromosome number, feature start and end
    "order": [[5, "asc"], [6, "asc"], [7, "asc"]],
    scrollY: $(window).height() - 330,
    scrollCollapse: true,
    paging: false,
    //info: false,
    'autoWidth': true

});

// define 2nd table similarly
cols = colsBase.slice();
for (var i = 0; i < table2_cols.length; i++) {
cols.push({title: table2_cols[i], "data": table2_cols[i], "type": "natural"})
}
var table2 = $('#table2').DataTable({
    ajax: table2_loc,
    columns: cols,
    "order": [[5, "asc"], [6, "asc"], [7, "asc"]],
    scrollY: $(window).height() - 330,
    scrollCollapse: true,
    paging: false,
    //info: false,
    'bAutoWidth': false
});

// display correlations of a clicked row as a sub-table
function format(d) {
    // get corr filters
    var min = $('#corrFilterSlider').data('slider').getValue()[0];
    var max = $('#corrFilterSlider').data('slider').getValue()[1];

    // d is the original data object for the row
    var table = ''
    table += '<table id="subtable" cellpadding="5" class="table table-striped table-bordered" cellspacing="0" border="0" >' +
        '<thead><tr>' +
        '<th>Correlated feature</th>' +
        '<th>R value</th>' +
        '<th>P value</th>' +
        '</thead></tr>'
    var names = d.CorrMapperCorrName.split('__|__');
    var r_vals = d.CorrMapperRval.split('__|__');
    var p_vals = d.CorrMapperPval.split('__|__');
    for (var i = 0; i < names.length; i++) {
        var r_val = parseFloat(r_vals[i]);
        // only show correlations that are between the slider ranges of #corrFilterSlider
        if (r_val >= min && r_val <= max) {
            table += '<tr data-value=' + names[i] + '>' +
                '<td>' + names[i] + '</td>' +
                '<td>' + r_vals[i] + '</td>' +
                '<td>' + p_vals[i] + '</td>' +
                '</tr>'
        }
    }
    table += '</table>';
    return table;
}

// ----------------------------------------------------------------------------
// DEFINE MOUSE EVENTS FOR TABLES AND SCATTER PLOTS
// ----------------------------------------------------------------------------

$('#table1 tbody').on('mouseover', 'tr', tableMouseOverRow);
$('#table1 tbody').on('mouseout', 'tr', tableMouseOutRow);
$('#table1 tbody').on('click', 'tr', tableClickedRow);

$('#table2 tbody').on('mouseover', 'tr', tableMouseOverRow);
$('#table2 tbody').on('mouseout', 'tr', tableMouseOutRow);
$('#table2 tbody').on('click', 'tr', tableClickedRow);

// hide scatters button
var show_scatter = true;
$('#scatter-toggle').change(function() {
    show_scatter = !$(this).prop('checked');
});

// function to get the scatter plot
function get_png_name(f1, f2){
    // takes in two feature names and converts it to a commutative filename
    var lf1 = f1.length;
    var lf2 = f2.length;
    // make then equal length
    if (lf1 > lf2){
        f2 = f2.concat(Array(lf1-lf2+1).join("a"));
    }else if(lf2 > lf1){
        f1 = f1.concat(Array(lf2-lf1+1).join("a"));
    }
    var filename = "";
    for (var i = 0; i < f1.length; i++) {
        var merged_char = (f1.charCodeAt(i) + f2.charCodeAt(i)).toString();

        filename = filename.concat(merged_char);
    }
    return(filename.concat(".png"));
}

// table row mouse over-out functions
function tableMouseOverRow(event) {
    var tr = $(this).closest('tr');
    var tableName = $(this).closest('table')[0].id;
    var table;
    if (tableName === 'table1') {
        table = table1;
    } else {
        table = table2;
    }
    var row = table.row(tr);
    //only highlight rows and network from main tables not subtables
    if (tableName !== 'subtable' && row.data()) {
        // highlight row
        tr.addClass('info');
        // highlight network
        networkMouseOver(row.data().CorrMapperBin);
        d3.select("#tooltip").classed("hidden", true);
    }else{
        if (tr.data("value") !== undefined && show_scatter){
            var feat1 = tr.data("value");
            var feat2 = currClickedFeature;
            var img_name = get_png_name(feat1, feat2);
            var out = "<img src='" + img_folder + img_name + "'>";
            // first set the position of the tooltip
            d3.select("#tooltip")
                .style("left", (event.clientX + 15) + "px")
                .style("top", (event.clientY - 80) + "px")
                .select("#value")
                .html(out);
            //Show the tooltip
            d3.select("#tooltip").classed("hidden", false);
        }
    }
}

function tableMouseOutRow() {
    // find our table and define basic vars
    var table = $(this).closest('table')[0].id;
    var opened;
    if (table === 'table1') {
        opened = openedTable1;
    } else {
        opened = openedTable2;
    }
    var tr = $(this).closest('tr');
    // if opened is defined check if we are rolling out from the opened row
    if (opened) {
        if (opened[0]._DT_RowIndex !== tr[0]._DT_RowIndex) {
            tr.removeClass('info');
        }
    } else {
        tr.removeClass('info');
    }
    networkMouseOut()
    d3.select("#tooltip").classed("hidden", true);
}

// table row click function
var openedTable1;
var openedTable2;
var currClickedFeature;
function tableClickedRow() {
    // find our table and define basic vars
    var tableName = $(this).closest('table')[0].id;
    var table, otherTable;
    if (tableName === 'table1') {
        table = table1;
        otherTable = table2;
    } else {
        table = table2;
        otherTable = table1;
    }

    // get other elements
    var tr = $(this).closest('tr');
    var row = table.row(tr);
    var data = row.data();
    currClickedFeature = data.Name;

    // open sub-table of correlated features
    if (!row.child.isShown()) {
        // show subtable of clicked row, save highlight of the network
        row.child(format(data)).show();
        networkClicked = data.CorrMapperBin;
        networkMouseOver(data.CorrMapperBin);
        // deal with previously clicked rows and subtables
        if (tableName === 'table1') {
            // update table title
            $("#tab2").text(data.Name);
            //close previously opened subtables in table1
            if (openedTable1) {
                table1.row(openedTable1).child.hide();
                openedTable1.removeClass('info');
            }
            openedTable1 = tr;
            // close subtables in table2 so we can filter it
            if (openedTable2) {
                table2.row(openedTable2).child.hide();
                openedTable2.removeClass('info');
                openedTable2 = '';
            }
        } else {
            // update table title
            $("#tab1").text(data.Name);
            //close previously opened subtables in table2
            if (openedTable2) {
                table2.row(openedTable2).child.hide();
                openedTable2.removeClass('info');
            }
            openedTable2 = tr;
            // close subtables in table1 so we can filter it
            if (openedTable1) {
                table1.row(openedTable1).child.hide();
                openedTable1.removeClass('info');
                openedTable1 = '';
            }
        }
    } else {
        // user clicked a previously opened subtable again in the same table
        if (tableName === 'table1') {
            table1.row(openedTable1).child.hide();
            openedTable1 = '';
        } else {
            table2.row(openedTable2).child.hide();
            openedTable2 = '';
        }
    }

    // clear previous filters on other table
    otherTable.search('').columns().search('').draw();
    // filter the other table to clicked element and sort by name
    otherTable.column(1).search(data.Name).draw();
    otherTable.order([4, 'asc']).draw();
}

// ----------------------------------------------------------------------------
// ADD DYNAMIC COLUMN TOGGLE AND RESET BUTTON TO SEARCH BAR
// ----------------------------------------------------------------------------

// readjust col widths of search bar

function addToggleBar(tableName) {
    $('#' + tableName + '_wrapper .col-sm-6:nth-child(1)')
        .removeClass('col-sm-6')
        .addClass('col-sm-9')
        .addClass('togglebar');
    $('#' + tableName + '_wrapper .col-sm-6:nth-child(2)')
        .removeClass('col-sm-6')
        .addClass('col-sm-3');

    // add reset button and build toogle menu from table_cols array
    var toggleMenu = '<p>Toggle column: ';
    var table, table_cols;
    if (tableName === 'table1'){
        table = table1;
        table_cols = table1_cols;
    }else{
        table = table2;
        table_cols = table2_cols;
    }
    for (var i = 0; i < table_cols.length; i++) {
        var d;
        if (table_cols[i] !== '# links') {
            d = table_cols[i];
        } else {
            d = "# links";
        }
        toggleMenu += '<a class="toggle-vis" data-column="' + (i + 4) + '">' + d + '</a>'
        if (i < table_cols.length - 1) {
            toggleMenu += '&nbsp;<span style="color: #aaaaaa">|</span>&nbsp;';
        } else {
            toggleMenu += '</p>';
        }
    }

    // add menu next to search bar and force hand cursor
    $(toggleMenu).appendTo('#' + tableName +'_wrapper .col-sm-9')
        .css("cursor", "pointer");

    // register click events for toggle menu items
    $('a.toggle-vis').on('click', function (e) {
        e.preventDefault();
        var column = table.column($(this).attr('data-column'));
        column.visible(!column.visible());
    });
}

// add toggle bar and reset button for both tables
addToggleBar('table1');
addToggleBar('table2');
$('#reset').click(resetTables);

function resetTables(){
    // reset clicked network element as well
    networkClicked = '';
    networkMouseOut();

    // reset tab titles
    if (data_file == "dataset1_2"){
        $("#tab1").text(table1_name);
        $("#tab2").text(table2_name);
    } else if(data_file == "dataset1"){
        $("#tab1").text(table1_name);
        $("#tab2").text(table1_name);
    } else if(data_file == "dataset2"){
        $("#tab1").text(table2_name);
        $("#tab2").text(table2_name);
    }

    // reset tables
    table1.search('').columns().search('').draw();
    table1.order([[5, "asc"], [6, "asc"], [7, "asc"]]).draw()
    table2.search('').columns().search('').draw();
    table2.order([[5, "asc"], [6, "asc"], [7, "asc"]]).draw()
    if (openedTable1) {
        openedTable1.removeClass('info');
        table1.row(openedTable1).child.hide();
        openedTable1.removeClass('shown');
    }
    if (openedTable2) {
        openedTable2.removeClass('info');
        table2.row(openedTable2).child.hide();
        openedTable2.removeClass('shown');
    }
    openedTable1 = '';
    openedTable2 = '';
}

// ----------------------------------------------------------------------------
// DEFINE TAB CHANGE AND KEY LISTENERS

// readjust columns after tab change
$('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
    $.fn.dataTable.tables({visible: true, api: true}).columns.adjust();
});

// Navigate between tabs with left and right keys
$("body").keydown(function (e) {
    if (e.keyCode == 37) { // left
        $('.nav-tabs a[href="#tab-table1"]').tab('show');
    }
    else if (e.keyCode == 39) { // right
        $('.nav-tabs a[href="#tab-table2"]').tab('show');
    }
    else if (e.keyCode == 27) { // Esc
        resetTables();
    }
});