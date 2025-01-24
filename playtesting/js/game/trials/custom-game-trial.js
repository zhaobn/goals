var jsPsychGameTrial = (function(jspsych) {
    "use strict";
  
    const info = {
        name: 'game-trial',
        parameters: {
            // Define your parameters here
            game_duration: {
                type: jspsych.ParameterType.INT,
                default: null
            },
            // Add more parameters as needed
        }
    };

    class GameTrial {
        constructor(jsPsych) {
            this.jsPsych = jsPsych;
        }

        trial(display_element, trial) {
            // Set up your game trial here
            display_element.innerHTML = `
                <div class="game-container">
                    <div class="game-stimulus">
                        <!-- Your game implementation -->
                    </div>
                </div>
            `;

            // Add your game logic here

            // End trial example:
            // this.jsPsych.finishTrial({
            //     // Add data to save
            //     score: gameScore,
            //     rt: responseTime
            // });
        }
    }

    return GameTrial;
})(jsPsychModule); 