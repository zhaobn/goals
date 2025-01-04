var jsPsychSurveyGrid = (function (jspsych) {
    "use strict";
  
    const info = {
      name: "survey-grid",
      parameters: {
        /** The HTML string to be displayed before the grid */
        stimulus: {
            type: jspsych.ParameterType.HTML_STRING,
            pretty_name: "Stimulus",
            default: undefined,
        },
        /** Array containing size of the grid. */
        grid_size: {
            type: jspsych.ParameterType.INT,
            array: true,
            pretty_name: "Grid size",
            default: [10, 10],
        },
        /** Array containing the labels for the dropdowns. */
        choices: {
            type: jspsych.ParameterType.STRING,
            pretty_name: "Choices",
            default: undefined,
            array: true,
        },
        /** Array containing the labels from the teacher. */
        labels: {
            type: jspsych.ParameterType.ARRAY,
            pretty_name: "Labels",
            default: undefined,
            array: true
        },
        /** The HTML for creating button. Can create own style. 
         * Use the "%choice%" string to indicate where the label from the choices parameter should be inserted. */
        button_html: {
            type: jspsych.ParameterType.HTML_STRING,
            pretty_name: "Button HTML",
            default: `<button class="jspsych-btn" style="
                margin-top: 20px; margin-bottom:20px; position: relative;">
                %choice%</button>`,
            array: true,
        },
      },
    };
  
/**
 * survey-grid
 *
 * jsPsych plugin for displaying an organized grid of images
 * each with a dropdown menu
 * used CoPilot for some help with responsiveness of plugin
 * i.e., adding event listeners & rendering properly
 *
//  * @author {Maya Malaviya}
//  * @see {@link {documentation-url}}
//  */

class jsPsychSurveyGridPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
  
    trial(display_element, trial) {
      /////////////////////////////////////////////////////////
      //          JSPSYCH PLUGIN CODE FOR DISPLAY            //
      /////////////////////////////////////////////////////////
  
      // Initialize HTML
      var html = '<div id="jspsych-survey-grid-stimulus">' + trial.stimulus + '</div>';
      html += '<div class="jspsych-survey-grid-container" style="display: grid; grid-template-columns: repeat(' + trial.grid_size[1] + ', auto); justify-content: center; align-items: center;">';
    
      var dino_img_nums =  [0, 3, 6, 9, 12, 15, 18, 60, 63, 66, 69, 72, 75, 78, 120, 123, 126, 129, 132, 135, 138, 180, 183, 186, 189, 192, 195, 198, 240, 243, 246, 249, 252, 255, 258, 300, 303, 306, 309, 312, 315, 318, 360, 363, 366, 369, 372, 375, 378]
      for (var i = 0; i < trial.grid_size[0] * trial.grid_size[1]; i++) {
          const x = i % trial.grid_size[1];
          const y = Math.floor(i / trial.grid_size[1]);
          const labelIndex = trial.labels.findIndex(label => label.coords.x === x && label.coords.y === y);
  
          html += '<div class="jspsych-survey-grid-item">';
          html += `<div class="image-container"><img src="../../stimuli/2d-dino-pngs/d${dino_img_nums[i]}.png" style="max-width: 90px"></div>`; 
          
          if (labelIndex !== -1) {
              const label = trial.labels[labelIndex];
              html += `<div class="highlighted-label" style="background-color: #ffde64; font-weight: bold;">&nbsp;&nbsp;&nbsp;${trial.choices[label.revealed_label]}&nbsp;&nbsp;&nbsp;</div>`;
          } else {
              html += '<select id="jspsych-survey-grid-dropdown-' + i + '">';
              html += '<option value=" "> </option>';  // Default blank option
              for (var j = 0; j < trial.choices.length; j++) {
                  html += '<option value="' + trial.choices[j] + '">' + trial.choices[j] + '</option>';
              }
              html += '</select>';
          }
  
          html += '</div>';
      }
      html += '</div>';
  
      // Add slider for confidence rating
      html += '<div id="jspsych-survey-grid-confidence" style="margin-top: 20px; text-align: center;">';
      html += '<p>On a scale of 0 to 100, how confident are you in your categorization judgments?</p>';
      html += '<input type="range" id="jspsych-survey-grid-slider" min="0" max="100" value="50">';
      html += '<span id="jspsych-survey-grid-slider-value">50</span>';
      html += '</div>';
  
      // Display button
      html += '<div id="jspsych-survey-grid-button" style="text-align: center; margin-top: 20px;">';
      var str = trial.button_html.replace(/%choice%/g, "Continue");
      html += str + '</div>';
  
      // Render HTML
      display_element.innerHTML = html;
  
      // Disable continue button initially
      var continueButton = display_element.querySelector('#jspsych-survey-grid-button button');
      continueButton.disabled = true;
  
      // Add event listener to slider
      var sliderUsed = false;
      var slider = display_element.querySelector('#jspsych-survey-grid-slider');
      var sliderValue = display_element.querySelector('#jspsych-survey-grid-slider-value');
      display_element.querySelector('#jspsych-survey-grid-slider').addEventListener('input', function () {
          sliderUsed = true;
          sliderValue.textContent = slider.value;
          checkIfContinueEnabled();
      });
  
      // Add event listeners to dropdowns
      var dropdowns = display_element.querySelectorAll('select');
      dropdowns.forEach(function (dropdown) {
          dropdown.addEventListener('change', function () {
              checkIfContinueEnabled();
          });
      });
  
      // Function to check if all dropdowns have a selected value other than " " and the slider is used
      function checkIfContinueEnabled() {
          var allDropdownsSelected = true;
          dropdowns.forEach(function (dropdown) {
              if (dropdown.value === " ") {
                  allDropdownsSelected = false;
              }
          });
          continueButton.disabled = !(allDropdownsSelected && sliderUsed);
      }
  
      // Add event listener to continue button
      continueButton.addEventListener('click', function () {
          var responses = [];
          for (var i = 0; i < trial.grid_size[0] * trial.grid_size[1]; i++) {
              const x = i % trial.grid_size[1];
              const y = Math.floor(i / trial.grid_size[1]);
              const labelIndex = trial.labels.findIndex(label => label.coords.x === x && label.coords.y === y);
  
              if (labelIndex === -1) {
                  var dropdown = display_element.querySelector('#jspsych-survey-grid-dropdown-' + i);
                  var response = {
                      stimulus: 'd' + dino_img_nums[i],
                      category: dropdown.options[dropdown.selectedIndex].value
                  };
                  responses.push(response);
              }
          }
          var confidence = display_element.querySelector('#jspsych-survey-grid-slider').value;
          var trial_data = {
              grid_responses: responses,
              confidence: confidence
          };
          console.log(trial_data);
          jsPsych.finishTrial(trial_data);
      });
    }
  }
  jsPsychSurveyGridPlugin.info = info;
  
  return jsPsychSurveyGridPlugin;
  })(jsPsychModule);