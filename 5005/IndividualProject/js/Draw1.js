/*//////////////////////////////////////////////////////////	
//Introduction
///////////////////////////////////////////////////////////*/
function Draw1(){

	/*First disable click event on clicker button*/
	stopClicker();
		
	/*Show and run the progressBar*/
	runProgressBar(time=700*8);
		
	changeTopText(newText = "Embed Knowledge to Chord Diagram",
	loc = 4/2, delayDisappear = 0, delayAppear = 1);

	changeBottomText(newText = "Jiawei ZHOU",
	loc = 1/2, delayDisappear = 0, delayAppear = 1);


	changeBottomText(newText = "Folk from Nadieh Bremer",
	loc = 1/2, delayDisappear = 3, delayAppear = 4);

	changeBottomText(newText = "",
	loc = 1/2, delayDisappear = 6, delayAppear = 8, finalText = true);

	changeTopText(newText = "Let's see how to do this",
	loc = 8/2, delayDisappear = 6, delayAppear = 8, finalText = true);
	

	
	//Remove arcs again
	d3.selectAll(".arc")
		.transition().delay(6*700).duration(2100)
		.style("opacity", 0)
		.each("end", function() {d3.selectAll(".arc").remove();});
		
};/*Draw1*/