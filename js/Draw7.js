/*//////////////////////////////////////////////////////////////////////////*/
//Nokia side of Samsung-Nokia chord
/*//////////////////////////////////////////////////////////////////////////*/
function Draw7(){

	/*First disable click event on clicker button*/
	stopClicker();
	/*Show and run the progressBar*/
	runProgressBar(time=700*9);	
	
	/*Samsung and Nokia text*/
	changeTopText(newText = "At the Nokia side, the arc is much thinner, only spanning 1.2%",
		loc = 4, delayDisappear = 0, delayAppear = 1);
	changeTopText(newText = "These 1.2% people switch from Samsung to Nokia",
		loc = 4, delayDisappear = 6, delayAppear = 7, finalText = true);
		
	/*Stop the color changing on the Samsung side*/
	d3.selectAll(".SamsungToNokiaArc")
		.transition().duration(700)
		.attr("fill", colors[5])
		.style("stroke", colors[5]);

	/*Repeatedly let an arc change colour*/		
	repeat();
	function repeat() {
		d3.selectAll(".NokiaToSamsungArc")
			.transition().duration(700)
			.attr("fill", "#99D2E9")
			.style('stroke', "#99D2E9")
			.transition().duration(700)
			.attr("fill", colors[4])
			.style("stroke", colors[4])
			.each("end", repeat)
			;
	};
				
};/*Draw7*/