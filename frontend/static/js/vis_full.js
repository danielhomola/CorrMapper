// ------------------------------------------------------------------------
// INTRO TO VIS
if (vis_intro == 1){
    $( document ).ready(help);
}

// LOAD THE DATASET
d3.csv(data_loc, dataReader, visualise);

function dataReader(d) {
    return {
        row:   +d.rowIdx,
        col:   +d.colIdx,
        r: +d.rVal,
        p: +d.pVal
    };
}

// ----------------------------------------------------------------------------
//
// MAIN FUNCTION
//
// ----------------------------------------------------------------------------

function visualise(error, data) {
    // DEFINING GLOBAL VARIABLES
    var margin = { top: 50, right: 20, bottom: 25, left: 55 };
    cellSize = 18,
    width = cellSize*colNumber,
    height = cellSize*rowNumber,
    fontMulti = 6.5,
    colorbarWidth = 50,
    legendXWidth = longestRow * fontMulti,
    legendYWidth = longestCol * fontMulti + margin.top,
    colorScale = d3.scale.linear()
        .domain([minData, 0, maxData])
        .range(["#0971B2", "#fff", "#CE1212"]);

    var totalWidth = colorbarWidth + width + margin.left + margin.right + legendXWidth ;
    var totalHeight = height + margin.bottom + legendYWidth;
    var heatmapWidth = d3.select('#heatmap').node().getBoundingClientRect().width;
    var scaler = totalWidth/heatmapWidth;

    // basics of the svg
    var svg = d3.select("#heatmap")
        .append("svg")
        // this is to make the heatmap responsive
        //.attr("width", "100%")
        .attr("width", totalWidth)
        //.attr("height", "100%")
        .attr("height", totalHeight)
        //.attr("preserveAspectRatio", "xMinYMin meet")
        //.attr("viewBox", "0 0 "+totalWidth*1.2+" " + totalHeight*1.2)

        .append("g")
        .attr("transform", "translate(" + margin.left + "," + legendYWidth + ")");

    // colorbar
    var colorbar = Colorbar()
        .origin([25,0])
        .scale(colorScale).barlength(height).thickness(15)
        .orient("vertical");

    var bar =  svg.append("g")
        .attr("class",".colorbar")
        .call(colorbar);

    // focus on network node once a label is clicked or hovered
    labelFocusNetwork = function (label, i) {
        network.focus(label[i]);
        network.draw();
        rings.focus(label[i]);
        rings.draw();
        selectedNetworkEl = label[i];
        noOverlayFocus();
    };

    // undoes the above
    resetLabelFocusNetwork = function(){
        if (selected.length === 0){
            selectedNetworkEl = '';
            drawNetworks(network_data, positions, connections, network, 'network', networkHeight);
            drawNetworks(network_data, positions, connections, rings, 'rings', ringsHeight);
        }else{
            filterNetwork(selected);
        }
        //drawNetworks(network_data, positions, connections, network, 'network', networkHeight);
        //drawNetworks(network_data, positions, connections, rings, 'rings', ringsHeight);
    };

    // ------------------------------------------------------------------------
    // ROW LABELS
    // ------------------------------------------------------------------------
    var featClicked=false;
    var featFocused=false;
    var rowSortOrder=false;
    var mouseOverTimeOutRow;
    var mouseOverTimeOutFlag = 0;
    var rowLabels = svg.append("g")
        .selectAll(".rowLabelg")
        .data(rowLabelLong)
        .enter()
        .append("text")
        .text(function (d) { return d; })
        .attr("x", colorbarWidth + cellSize*colNumber)
        .attr("y", function (d, i) { return i * cellSize; })
        .style("text-anchor", "start")
        .style("font-size","13px")
        .attr("transform", "translate(15," + cellSize / 1.5 + ")")
        .attr("class", function (d,i) { return "rowLabel mono r"+i;} )
        .on("mouseover", function(d, i) {
            d3.select(this).classed("text-hover",true);
            mouseOverTimeOutRow = setTimeout(function(){
                d3.selectAll(".colLabel").classed("text-highlight",false);
                d3.selectAll(".rowLabel").classed("text-highlight",false);
                featClicked=false;
                featFocused=true;
                mouseOverTimeOutFlag = 1;
                labelFocusNetwork(rowLabel, i);
            }, 500);
        })
        .on("mouseout" , function(d) {
            d3.select(this).classed("text-hover",false);
            clearTimeout(mouseOverTimeOutRow);
            if(mouseOverTimeOutFlag === 1){
                mouseOverTimeOutFlag = 0;
                if (!featClicked) {
                    resetLabelFocusNetwork();
                }
                featFocused=false;
            }
        })
        .on("click", function(d,i) {
            rowSortOrder=!rowSortOrder;
            sortByLabel("r",i,rowSortOrder);
            if (!featFocused) {
                labelFocusNetwork(rowLabel, i);
            }
            featClicked=true;
            d3.selectAll(".colLabel").classed("text-highlight",false);
            d3.selectAll(".rowLabel").classed("text-highlight",false);
            d3.select(this).classed("text-highlight",true);

            // remove the col highlight in the reorder menu and hide modules
            $("#reorder-dropdown .active.colorder").removeClass("active");
            $("#modules-dropdown-main").hide();
            $("#modules-dropdown .active").removeClass("active");
        });

    // ------------------------------------------------------------------------
    // COL LABELS
    // ------------------------------------------------------------------------

    var colSortOrder=false;
    var mouseOverTimeOutCol;
    var colLabels = svg.append("g")
        .selectAll(".colLabelg")
        .data(colLabelShort)
        .enter()
        .append("text")
        .text(function (d) { return d; })
        .attr("x", 0)
        .attr("y", function (d, i) { return colorbarWidth + i * cellSize; })
        .style("text-anchor", "left")
        .style("font-size","13px")
        .attr("transform", "translate("+cellSize/1.5 + ",-6) rotate (-90)")
        .attr("class",  function (d,i) { return "colLabel mono c"+i;} )
        .on("mouseover", function(d, i) {
            d3.select(this).classed("text-hover",true);
            mouseOverTimeOutCol = setTimeout(function(){
                d3.selectAll(".colLabel").classed("text-highlight",false);
                d3.selectAll(".rowLabel").classed("text-highlight",false);
                featClicked=false;
                featFocused=true;
                mouseOverTimeOutFlag = 1;
                labelFocusNetwork(colLabel, i);
            }, 500);
        })
        .on("mouseout" , function(d) {
            d3.select(this).classed("text-hover",false);
            clearTimeout(mouseOverTimeOutCol);
            if(mouseOverTimeOutFlag === 1){
                mouseOverTimeOutFlag = 0;
                if (!featClicked) {
                    resetLabelFocusNetwork();
                }
            }
        })
        .on("click", function(d,i) {
            colSortOrder=!colSortOrder;
            sortByLabel("c",i,colSortOrder);
            if (!featFocused) {
                labelFocusNetwork(colLabel, i);
            }
            featClicked=true;
            d3.selectAll(".colLabel").classed("text-highlight",false);
            d3.selectAll(".rowLabel").classed("text-highlight",false);
            d3.select(this).classed("text-highlight",true);

            // remove the row highlight in the reorder menu and hide modules
            $("#reorder-dropdown .active.roworder").removeClass("active");
            $("#modules-dropdown-main").hide();
            $("#modules-dropdown .active").removeClass("active");
        });

    // ------------------------------------------------------------------------
    //
    // HEATMAP
    //
    // ------------------------------------------------------------------------

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

    var heatMap = svg.append("g").attr("class","g3")
            .selectAll(".cellg")
            .data(data,function(d){return d.row+":"+d.col;})
            .enter()
            .append("rect")
            .attr("x", function(d) { return colorbarWidth + defaultColOr.indexOf(d.col) * cellSize; })
            .attr("y", function(d) { return defaultRowOr.indexOf(d.row) * cellSize; })
            .attr("class", function(d){return "cell cell-border cr"+(d.row)+" cc"+(d.col);})
            .attr("width", cellSize)
            .attr("height", cellSize)
            .style("fill", function(d) { return colorScale(d.r); })
            .on("mouseover", function(d){
                // colorbar pointer
                bar.pointTo(d.r);

                //highlight text
                d3.select(this).classed("cell-hover",true);
                d3.selectAll(".rowLabel").classed("text-highlight",function(r,ri){ return ri==(d.row);});
                d3.selectAll(".colLabel").classed("text-highlight",function(c,ci){ return ci==(d.col);});

                //Update the tooltip wth new image
                if (rowLabelShort[d.row] !== colLabelShort[d.col] && d.r !== 0 && show_scatter) {

                    /* - old toolbox for displaying info about edges
                    var out = "Column: " + colLabelShort[d.col];
                    if (colFoldChange[d.col] !== '') {
                        out += " &nbsp|&nbspFC: " + colFoldChange[d.col];
                    }
                    out += "<br/>Row: " + rowLabelShort[d.row];
                    if (rowFoldChange[d.row] !== '') {
                        out += "&nbsp|&nbspFC: " + rowFoldChange[d.row];
                    }
                    out += "<br/>R<sup>2</sup>: " + d.r + "<br/>p-value</sup>: " + d.p;
                    */

                    // test both file names: colname_rowname and rowname_colname
                    var img_name = get_png_name(rowLabelShort[d.row], colLabelShort[d.col]);
                    var out = "<img src='" + img_folder + img_name + "'>";

                    // first set the position of the tooltip
                    d3.select("#tooltip")
                        .style("left", (d3.event.pageX + 15) + "px")
                        .style("top", (d3.event.pageY - 80) + "px")
                        .select("#value")
                        .html(out);
                    //Show the tooltip
                    d3.select("#tooltip").classed("hidden", false);

                }
            })

            .on("mouseout", function(){
                d3.select(this).classed("cell-hover",false);
                d3.selectAll(".rowLabel").classed("text-highlight",false);
                d3.selectAll(".colLabel").classed("text-highlight",false);
                d3.select("#tooltip").classed("hidden", true);
            });

    // ------------------------------------------------------------------------
    //
    // HEATMAP SORTING
    //
    // ------------------------------------------------------------------------

    function sortByLabel(rowOrCol,i,sortOrder){
        var t = svg.transition().duration(1500);
        var valsToReorder=[];
        var sorted;
        d3.selectAll(".c"+rowOrCol+i)
            .filter(function(d){
                valsToReorder.push(d.r);
            });

        // deselect everything, reset network
        d3.selectAll(".cell-selected").classed("cell-selected",false);
        d3.selectAll(".rowLabel").classed("text-selected",false);
        d3.selectAll(".colLabel").classed("text-selected",false);
        if (selected.length !== 0){
          resetSelection();
        }

        // SORT BY CLICKING ROW
        if(rowOrCol=="r"){
            sorted=d3.range(colNumber).sort(function(a,b){
                if(sortOrder){
                    return valsToReorder[b]-valsToReorder[a];}
                else{
                    return valsToReorder[a]-valsToReorder[b];}});
            t.selectAll(".cell")
                .attr("x", function(d) { return colorbarWidth + sorted.indexOf(d.col) * cellSize; });
            t.selectAll(".colLabel")
                .attr("y", function (d, i) { return colorbarWidth + sorted.indexOf(i) * cellSize; });

        // SORT BY CLICKING COLUMN
        }else{
            sorted=d3.range(rowNumber).sort(function(a,b){
                if(sortOrder){
                    return valsToReorder[b]-valsToReorder[a];}
                else{
                    return valsToReorder[a]-valsToReorder[b];}});
            t.selectAll(".cell")
                .attr("y", function(d) { return sorted.indexOf(d.row) * cellSize; });
            t.selectAll(".rowLabel")
                .attr("y", function (d, i) { return sorted.indexOf(i) * cellSize; });
        }
    }

    // ------------------------------------------------------------------------
    // REORDER HEATMAP WITH DROP DOWN MENU
    // ------------------------------------------------------------------------

    $("#reorder-dropdown li").on("click", function () {
        var id = $(this).data('value');
        var to_declass = "row";
        switch(id) {
            // rows
            case "defaultRow":
                reorder("row", defaultRowOr);
                break;
            case "dataRow":
                reorder("row", dataRowOr);
                break;
            case "rRow":
                reorder("row", rRowOr);
                break;
            case "nameRow":
                reorder("row", nameRowOr);
                break;
            // cols
            case "defaultCol":
                reorder("col", defaultColOr);
                to_declass = "col";
                break;
            case "dataCol":
                reorder("col", dataColOr);
                to_declass = "col";
                break;
            case "rCol":
                reorder("col", rColOr);
                to_declass = "col";
                break;
            case "nameCol":
                reorder("col", nameColOr);
                to_declass = "col";
                break;
        }
        // deselect previously selected row or col dropdown element
        if (to_declass === "row"){
            $("#reorder-dropdown .active.roworder").removeClass("active");
        }else{
            $("#reorder-dropdown .active.colorder").removeClass("active");
        }
        $(this).addClass("active");

        // only show the module selector if both rows and cols are ordered by it
        var actives = ["", ""];
        $("#reorder-dropdown .active").each(function (i){
            actives[i] = $(this).data('value');
        });
        if (actives.indexOf("defaultRow") !== -1 && actives.indexOf("defaultCol") !== -1){
            $("#modules-dropdown-main").show();
        }else{
            $("#modules-dropdown-main").hide();
            $("#modules-dropdown .active").removeClass("active");
        }
    });

    function reorder(axis, order){
        // deselect everything, reset network
        d3.selectAll(".cell-selected").classed("cell-selected",false);
        d3.selectAll(".rowLabel").classed("text-selected",false);
        d3.selectAll(".colLabel").classed("text-selected",false);
        resetSelection();

        var t = svg.transition().duration(1500);
        if (axis === 'row'){
            t.selectAll(".cell")
                .attr("y", function(d) { return order.indexOf(d.row) * cellSize; });
            t.selectAll(".rowLabel")
                .attr("y", function (d, i) { return order.indexOf(i) * cellSize; });
        } else {
            t.selectAll(".cell")
                .attr("x", function(d) { return colorbarWidth + order.indexOf(d.col) * cellSize; });
            t.selectAll(".colLabel")
                .attr("y", function (d, i) { return colorbarWidth + order.indexOf(i) * cellSize; });
        }
    }


    // ------------------------------------------------------------------------
    //
    // EXTRA NAVIGATON FUNCTIONS
    //
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // COPY TO CLIPBOARD
    // ------------------------------------------------------------------------

    function getSelectedFeatures(){
        // update selected feature array
        selectedFeatures = [];
        d3.selectAll('.cell-selected').filter(function(cell_d, i) {
            if(cell_d.r !== 0){
                var row_feat = rowLabelLong[cell_d.row];
                var col_feat = colLabelLong[cell_d.col];
                if (selectedFeatures.indexOf(row_feat) === -1){
                    selectedFeatures.push(row_feat);
                }
                if (selectedFeatures.indexOf(col_feat) === -1){
                    selectedFeatures.push(col_feat);
                }
            }
        });
    }

    var clipboard = new Clipboard('#to_clipboard', {
        text: function(trigger) {
            var to_clipboard;
            if (selectedFeatures.length === 0){
                to_clipboard = "No non-zero cells were selected!";
            }else{
                to_clipboard = selectedFeatures.join(", ");
            }
            return to_clipboard
        }
    });

    // once clicked show to user that everything is the clipboard
    clipboard.on('success', function(e) {
        $('#to_clipboard a').text("Copied!");

        var copyTimer;
        clearTimeout(copyTimer);
        copyTimer = setTimeout(function () {
            $('#to_clipboard a').text("Copy to clipboard");
        }, 1500);
    });
    // hide by default
    $('#to_clipboard').hide();

    // ------------------------------------------------------------------------
    // SEARCHBAR
    // ------------------------------------------------------------------------

    $('#searchbar').on('input', search_function)
    function search_function() {
        var search_term = $('#searchbar').val().toLowerCase();
            colLabelLong.forEach(function (v, i) {
                v = v.toLowerCase();
                if (v.indexOf(search_term) !== -1 && search_term !== "") {
                    d3.select(".c" + (i) + ".colLabel").classed("text-search", true);
                } else {
                    d3.select(".c" + (i) + ".colLabel").classed("text-search", false);
                }
            });

            rowLabelLong.forEach(function (v, i) {
                v = v.toLowerCase();
                if (v.indexOf(search_term) !== -1 && search_term !== "") {
                    d3.select(".r" + (i) + ".rowLabel").classed("text-search", true);
                } else {
                    d3.select(".r" + (i) + ".rowLabel").classed("text-search", false);
                }
            });
    }

    // ------------------------------------------------------------------------
    // HIDE SCATTER PLOTS
    // ------------------------------------------------------------------------

    var show_scatter = true;
    $('#scatter-toggle').change(function() {
        show_scatter = !$(this).prop('checked');
    });

    // ------------------------------------------------------------------------
    //
    // MULTI SELECTION & MODULE SELECTION
    //
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // HIGHLIGHT MODULES WITH DROPDOWN
    // ------------------------------------------------------------------------

    $("#modules-dropdown li").on("click", function () {
        var id = $(this).data('value');
        selected = modules[id];
        filterNetwork(selected);

        // move active menu rollover to the selected subitem
        $("#modules-dropdown .active").removeClass("active");
        $(this).addClass("active");

        // delete any active selection
        d3.selectAll(".cell-selected").classed("cell-selected",false);
        d3.selectAll(".rowLabel").classed("text-selected",false);
        d3.selectAll(".colLabel").classed("text-selected",false);

        // find the cells and labels that need to be highlighted
        selected.split(",").forEach(function(v,i){
            var rc = v.split("|");
            // if we have sym matrix, selected variable calls cols rows for the network
            // but we need cols to be cols for the heatmap and labels still.
            var col_ix = rc[1];
            if (data_file !== "dataset1_2"){
                col_ix = col_ix.replace("r", "c");
            }
            var class_of_cell = ".c" + rc[0] + ".c" + col_ix;
            var class_of_rowlabel = ".rowLabel." + rc[0];
            var class_of_collabel = ".colLabel." + col_ix;
            d3.select(class_of_cell).classed("cell-selected",true);
            d3.select(class_of_rowlabel).classed("text-selected",true);
            d3.select(class_of_collabel).classed("text-selected",true);
        });

        // show copy to clipboard
        getSelectedFeatures();
        $("#to_clipboard").show();
    });


    // ------------------------------------------------------------------------
    // MULTI SELECTION
    // ------------------------------------------------------------------------

    // collects cells as we drag the mouse
    var currSelected = [];
    // collects cells of final selection
    var selected = [];
    // names of selected features with non zero value in final selection
    var selectedFeatures = [];
    var sa=d3.select(".g3")
            .on("mousedown", function() {
                currSelected = [];
                if( !d3.event.shiftKey) {
                    selected = [];
                    selectedFeatures = [];
                    d3.selectAll(".cell-selected").classed("cell-selected",false);
                    d3.selectAll(".rowLabel").classed("text-selected",false);
                    d3.selectAll(".colLabel").classed("text-selected",false);
                    // deselect any active module in drop down
                    $("#modules-dropdown .active").removeClass("active");
                }
                var p = d3.mouse(this);
                sa.append("rect")
                    .attr({
                        rx      : 0,
                        ry      : 0,
                        class   : "selection",
                        x       : p[0],
                        y       : p[1],
                        width   : 1,
                        height  : 1
                    })
            })
            .on("mousemove", function() {
                var s = sa.select("rect.selection");

                if(!s.empty()) {
                    var p = d3.mouse(this),
                        d = {
                            x       : parseInt(s.attr("x"), 10),
                            y       : parseInt(s.attr("y"), 10),
                            width   : parseInt(s.attr("width"), 10),
                            height  : parseInt(s.attr("height"), 10)
                        },
                        move = {
                            x : p[0] - d.x,
                            y : p[1] - d.y
                        };
                    if(move.x < 0 || (move.x*2<d.width)) {
                        d.x = p[0];
                        d.width -= move.x;
                    } else {
                        d.width = move.x;
                    }
                    if(move.y < 0 || (move.y*2<d.height)) {
                        d.y = p[1];
                        d.height -= move.y;
                    } else {
                        d.height = move.y;
                    }
                    s.attr(d);

                    // deselect all temporary selected state objects
                    d3.selectAll('.cell-selection.cell-selected').classed("cell-selected", false);
                    d3.selectAll(".text-selection.text-selected").classed("text-selected",false);

                    var cell_ind;
                    d3.selectAll('.cell').filter(function(cell_d, i) {
                        if(
                            !d3.select(this).classed("cell-selected") &&
                            // inner circle inside selection frame
                            (this.x.baseVal.value)+cellSize >= d.x && (this.x.baseVal.value)<=d.x+d.width &&
                            (this.y.baseVal.value)+cellSize >= d.y && (this.y.baseVal.value)<=d.y+d.height
                        ) {
                            d3.select(this)
                                .classed("cell-selection", true)
                                .classed("cell-selected", true);
                            d3.select(".r"+(cell_d.row))
                                .classed("text-selection",true)
                                .classed("text-selected",true);
                            d3.select(".c"+(cell_d.col))
                                .classed("text-selection",true)
                                .classed("text-selected",true);

                            // if we only have one dataset the col and row names are the same
                            if (data_file !== "dataset1_2"){
                                cell_ind = "r" + cell_d.row + "|" +  "r" + cell_d.col;
                            }else {
                                cell_ind = "r" + cell_d.row + "|" + "c" + cell_d.col;
                            }
                            if (currSelected.indexOf(cell_ind) === -1){
                                currSelected.push(cell_ind)
                            }
                        }else{
                            // remove selected cells from the list which the user hover over but did not select in the end
                            if (data_file !== "dataset1_2"){
                                cell_ind = "r" + cell_d.row + "|" +  "r" + cell_d.col;
                            }else {
                                cell_ind = "r" + cell_d.row + "|" + "c" + cell_d.col;
                            }
                            var toRemove = currSelected.indexOf(cell_ind);
                            if (toRemove > -1) {
                                currSelected.splice(toRemove, 1)
                            }
                        }
                    });
                }
            })
            .on("mouseup", function() {
                // remove selection frame
                sa.selectAll("rect.selection").remove();

                // remove temporary selection marker class
                d3.selectAll('.cell-selection').classed("cell-selection", false);
                d3.selectAll(".text-selection").classed("text-selection",false);

                // update networks by selection
                selected = selected.concat(currSelected);

                getSelectedFeatures();

                // show copy clipboard
                if (selectedFeatures.length > 0){
                    $('#to_clipboard').show();
                }
                if (selected.length === 0){
                    $('#to_clipboard').hide();
                    selectedNetworkEl = '';
                    drawNetworks(network_data, positions, connections, network, 'network', networkHeight);
                    drawNetworks(network_data, positions, connections, rings, 'rings', ringsHeight);
                }else{
                    filterNetwork(selected);
                }
            })
            .on("mouseout", function() {
                if(d3.event.relatedTarget.tagName=='html') {
                    // remove selection frame
                    sa.selectAll("rect.selection").remove();
                    // remove temporary selection marker class
                    d3.selectAll('.cell-selection').classed("cell-selection", false);
                    d3.selectAll(".rowLabel").classed("text-selected",false);
                    d3.selectAll(".colLabel").classed("text-selected",false);
                }
            });

    // ------------------------------------------------------------------------
    //
    // NETWORK VISUALISATION
    //
    // ------------------------------------------------------------------------

    // this holds the current focus of the networks
    var selectedNetworkEl = '';

    var network = d3plus.viz().container("#network");
    var networkHeight;
    if (two_column == 1){
        networkHeight = $(window).height() - 300
    }else{
        networkHeight = 700
    }
    drawNetworks(network_data, positions, connections, network, 'network', networkHeight);

    var rings = d3plus.viz().container("#rings");
    var ringsHeight = 700;
    drawNetworks(network_data, positions, connections, rings, 'rings', ringsHeight);

    // main network drawing function

    function drawNetworks(data, positions, connections, container, type, height) {
        var size;
        if (type === 'network'){
            size = 13;
        }else{
            size = {
                "scale":d3.scale.identity()
            }
        }

        container
            .type(type)
            .height(height)
            .data(data)
            .id("name")
            .nodes({
                "overlap":.1,
                "value": positions
            })
            .edges({
                "size": "width",
                "color": function(e) {
                    return colorScale(e.rval);
                },
                "opacity":0.6,
                "value": connections
            })
            .text({
                "value":"shortLabel"
            })
            .tooltip({
                "small": 300,
                "anchor": "top right",
                "value": [
                        function(d){return d.longLabel},
                        "foldChange",
                        "R<sup>2<sup>"
                        ]})
            .color("color")
            .font({
                "family": "Arial"
            })
            .history(false)
            .size(size)
            .background("#fff")
            .focus(false)
            .zoom(true)
            .legend(false)
            .draw();
    }

    // ------------------------------------------------------------------------
    // NETWORK FILTERING
    // ------------------------------------------------------------------------

    // when we select a subset of heatmap cells we filter networks with this
    function filterNetwork(selected){
        // only keep nodes which are between edges with non-zero R^2
        var selected2 = [];
        var f3 = connections.filter(function (el) {
            var toCheck = el.source.name + '|' + el.target.name;
            var toCheck2 = el.target.name + '|' + el.source.name

            if (selected.indexOf(toCheck)>=0 || selected.indexOf(toCheck2)>=0){
                selected2.push(el.source.name);
                selected2.push(el.target.name);
                return true;
            };
        });
        selected2 = jQuery.unique(selected2);
        var f1 = network_data.filter(function (el) {
            return (selected2.indexOf(el.name)>=0)
        });

        var f2 = positions.filter(function (el) {
            return (selected2.indexOf(el.name)>=0)
        });
        if (f1.length > 0){
            drawNetworks(f1, f2, f3, network, 'network', networkHeight, networkHeight);
            drawNetworks(f1, f2, f3, rings, 'rings', ringsHeight, ringsHeight);
        }

    }

    // called from vis.html to reset all selection and focus
    $('#reset_network').click(resetSelection);
    $("body").keydown(function (e) {
        if (e.keyCode == 27) { // Esc
            resetSelection();
        }
    });
    function resetSelection(){
        // searchbar
        $('#searchbar').val("");
        search_function();
        // copy to clipboard
        $('#to_clipboard').hide();
        // deactivate all modules selected
        $("#modules-dropdown .active").removeClass("active");

        // selection and networks
        selected = [];
        selectedFeatures = [];
        d3.selectAll(".cell-selected").classed("cell-selected",false);
        d3.selectAll(".rowLabel").classed("text-selected",false);
        d3.selectAll(".colLabel").classed("text-selected",false);
        d3.selectAll(".rowLabel").classed("text-highlight",false);
        d3.selectAll(".colLabel").classed("text-highlight",false);
        selectedNetworkEl = '';
        drawNetworks(network_data, positions, connections, network, 'network', networkHeight);
        drawNetworks(network_data, positions, connections, rings, 'rings', ringsHeight);
        // reset zooms as well
        zoom_network.scale(1).translate([0, 0]);
        d3.select("#network #zoom").transition().duration(500)
            .attr("transform", "translate(" + zoom_network.translate() + ")scale(" + zoom_network.scale() + ")");
        zoom_rings.scale(1).translate([0, 0]);
        d3.select("#rings #zoom").transition().duration(500)
            .attr("transform", "translate(" + zoom_rings.translate() + ")scale(" + zoom_rings.scale() + ")");
    }

    // ------------------------------------------------------------------------
    // CUSTOM CLICK FUNCTIONS
    // hacks to get rid of the default hover and focus colours on edges
    // and get the focus sync working between the two networks
    // ------------------------------------------------------------------------

    network.mouse({
        'click':function(point){
            customClick(point)
        }
    });
    rings.mouse({
        'click':function(point){
            customClick(point)
        }
    });

    function noOverlayFocus(){
        // the logic here is that the only way we can identify the edges of the focused region is by
        // going through the edges in the #focus div, saving their x1, x2, y1, y2 coordinates and
        // fading all edges in #network > #edges that have different x1, x2, y1, y2 coordinates.
        setTimeout(function(){
            x1s = [];
            x2s = [];
            y1s = [];
            y2s = [];
            d3.selectAll("#focus").selectAll(".d3plus_edge_line").selectAll('line').filter(function(edge, i) {
                x1s.push(this.x1.baseVal.value);
                x2s.push(this.x2.baseVal.value);
                y1s.push(this.y1.baseVal.value);
                return true;
            });
            d3.selectAll("#network").selectAll("#edges").selectAll(".d3plus_edge_line").selectAll('line')
                .filter(function(edge, i) {
                    var x1 = this.x1.baseVal.value;
                    var x2 = this.x2.baseVal.value;
                    var y1 = this.y1.baseVal.value;
                    var y2 = this.y2.baseVal.value;
                    var x1ind = x1s.indexOf(x1);
                    var x2ind = x2s.indexOf(x2);
                    var y1ind = y1s.indexOf(y1);
                    var y2ind = y2s.indexOf(y2);
                    if (x1ind == x2ind && x1ind == y1ind && x1ind == y2ind){
                        d3.select(this).transition().duration(200).attr("opacity",.1);
                    }else{
                        d3.select(this).transition().duration(200).attr("opacity", 1);
                    }
                });
            // this would get decreased to .2 by the default click function
            d3.selectAll("#edges").transition().duration(100).attr("opacity", 1);
            // hide focus overlay lines
            d3.selectAll("#focus").selectAll(".d3plus_edge_line").selectAll('line')
                .transition().duration(100)
                .attr('opacity',.0);
        }, 100);
    }

    function customClick(point){
        if (point.name !== selectedNetworkEl || selectedNetworkEl === ''){
            // sync up focus events between networks
            selectedNetworkEl = point.name;
            network.focus(point.name);
            network.draw();
            rings.focus(point.name);
            rings.draw();
            // get rid off grey overlay on focus and selectively fade edges to show focused edges
            noOverlayFocus();
            // highlight heatmap cols and rows
            d3.selectAll(".colLabel").classed("text-highlight",false);
            d3.selectAll(".rowLabel").classed("text-highlight",false);
            var cl = "." + point.name;
            d3.select(cl).classed("text-highlight",true);
            // if we have one dataset, highlight both cols and rows of clicked feature
            if (data_file !== "dataset1_2") {
                cl = ".c" + point.name.slice(1, point.name.length);
                d3.select(cl).classed("text-highlight", true);
            }
        }else{
            selectedNetworkEl = '';
            network.focus(false);
            network.draw();
            rings.focus(false);
            rings.draw();
            d3.selectAll(".colLabel").classed("text-highlight",false);
            d3.selectAll(".rowLabel").classed("text-highlight",false);
        }
    }

    // ------------------------------------------------------------------------
    // CUSTOM OVER FUNCTIONS
    // ------------------------------------------------------------------------

    d3.select("#network").on("mouseover", customHover);
    d3.select("#rings").on("mouseover", customHover);

    function customHover(){
        // network
        d3.selectAll("#edge_hover").selectAll(".d3plus_edge_line").selectAll('line')
            .transition().duration(200)
            .style('stroke-width', 12).style('stroke','#000').attr('opacity',.2);
        // rings
        d3.selectAll("#edge_hover").selectAll(".d3plus_edge_path").selectAll('path')
            .transition().duration(200)
            .style('stroke-width', 12).style('stroke','#000').attr('opacity',.2);
    }

    // ------------------------------------------------------------------------
    // ADDING BACK ZOOM
    // ------------------------------------------------------------------------

    // since we use custom click and hover functions the default d3plus zooming
    // behaviour is gone so we need to rebuild it manually using d3
    var zoom_network = d3.behavior.zoom()
        .scaleExtent([1, 10])
        .on("zoom", zoomed_network);
    d3.select("#network").call(zoom_network);
    function zoomed_network() {
        d3.select("#network #zoom").attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }

    var zoom_rings = d3.behavior.zoom()
        .scaleExtent([1, 10])
        .on("zoom", zoomed_rings);
    d3.select("#rings").call(zoom_rings);
    function zoomed_rings() {
        d3.select("#rings #zoom").attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }


    // ------------------------------------------------------------------------
    // RESPONSIVENESS
    // ------------------------------------------------------------------------

    var resizeTimer;
    $(window).on('resize', function(e) {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function () {
            network
                .width(d3.select('#network').node().getBoundingClientRect().width)
                .draw();
            rings
                .width(d3.select('#rings').node().getBoundingClientRect().width)
                .draw();
        }, 100);
    });

};


