// Initialize jsPsych
const jsPsych = initJsPsych({
    on_finish: function() {
        // Handle experiment completion
        jsPsych.data.displayData();
    }
});

// Define your circle trial
const circleTrialProcedure = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="game-container">
                <div class="game-stimulus">
                    <svg width="200" height="200" viewBox="0 0 200 200">
                        <circle 
                            class="vector-circle"
                            cx="100" 
                            cy="100" 
                            r="80"
                        />
                    </svg>
                </div>
            </div>
        `;
    },
    choices: "ALL_KEYS",
    trial_duration: 2000 // Circle will display for 2 seconds
};

// Create the experiment timeline
const timeline = [
    // Add instructions, consent, etc.
    circleTrialProcedure,
    // Add post-game questionnaires, etc.
];

// Start the experiment
jsPsych.run(timeline); 