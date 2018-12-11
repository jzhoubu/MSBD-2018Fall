/*//////////////////////////////////////////////////////////////////////////*/
// Samsung net gain from Nokia
/*//////////////////////////////////////////////////////////////////////////*/
function Draw8(){

	/*First disable click event on clicker button*/
	stopClicker();
	/*Show and run the progressBar*/
	runProgressBar(time=700*11);	
	
	/*Samsung and Nokia text*/
	changeTopText(newText = "Since the Samsung is attracting more customers from Nokia than lossing them, the chord is the color of Samsung blue",
		loc = 5, delayDisappear = 0, delayAppear = 1, finalText = true);
		
	/*Stop the colour changing on the Nokia side*/
	d3.selectAll(".NokiaToSamsungArc")
		.transition().duration(700)
		.attr("fill", colors[4])
		.style("stroke", colors[4]);
				
};/*Draw8*/