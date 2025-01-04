var jsPsychGoalPursuit = (function (jspsych) {
    'use strict';
    
    const info = {
        name: "goal-pursuit",
        parameters: {
            instruction: {
                type: jspsych.ParameterType.STRING,
                default: "Select your goal and take actions.",
                description: "Instructions displayed above the task."
            },
            goalState: {
                type: jspsych.ParameterType.OBJECT,
                default: null,
                description: "Goal state passed from the builder"
            },
            participant_id: {
                type: jspsych.ParameterType.STRING,
                default: null,
                description: "Participant ID"
            }
        }
    };

    class GoalPursuitPlugin {
        constructor(jsPsych) {
            this.jsPsych = jsPsych;
            this.currentSelect = [];
            this.currentConfig = null;
            this.currentGoal = null;
            this.interactionCount = 0;
            this.trialParams = null;
            this.colorCode = {
                "1": "lightblue",
                "2": "blue",
                "3": "darkblue",
                "4": "purple",
                "5": "darkpurple"
            };
        }

        trial(display_element, trial) {
            this.startTime = Date.now();
            this.initialConfig = { ...this.currentConfig };
            this.interactionCount = 0;
            this.trialParams = trial;
            
            // HTML Structure
            const html = `
                <div class="container">
                    <div class="goal-display">
                        <h2 class="current-state">Current Goal</h2>
                        <div class="innerworkspace-top">
                            <div class="holder" id="fixed-holder-a"></div>
                        </div>
                        <div class="innerworkspace-bot">
                            <div class="holder" id="fixed-holder-b"></div>
                            <div class="holder" id="fixed-holder-c"></div>
                        </div>
                    </div>
                    <div class="pursuit-container">
                        <div class="workspace">
                            <h2 class="current-state">Current State</h2>
                            <div class="innerworkspace-top">
                                <div class="holder" id="demo-holder-a"></div>
                            </div>
                            <div class="innerworkspace-bot">
                                <div class="holder" id="demo-holder-b"></div>
                                <div class="holder" id="demo-holder-c"></div>
                            </div>
                        </div>
                        <button id="large-red-button" class="abandon-button">Abandon Goal</button>
                    </div>
                </div>
            `;

            display_element.innerHTML = html;

            // CSS Styling
            const styles = document.createElement("style");
            styles.innerHTML = `
                .container {
                    display: flex;
                    justify-content: center;
                    align-items: flex-start;
                    gap: 40px;
                    padding: 20px;
                    width: 100%;
                    max-width: 1200px;
                    margin: 0 auto;
                }

                .pursuit-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 20px;
                }

                .holder {
                    height: 150px;
                    width: 150px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 10px;
                    padding: 5px;
                    box-sizing: border-box;
                }

                .innerworkspace-top {
                    height: 35%;
                    width: 90%;
                    margin: 10px auto;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .innerworkspace-bot {
                    height: 35%;
                    width: 90%;
                    margin: 10px auto;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 20px;
                }

                .abandon-button {
                    padding: 10px 20px;
                    font-size: 16px;
                    background-color: #f0f0f0;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }

                .abandon-button:hover {
                    background-color: #e0e0e0;
                }

                .current-state {
                    font-size: 24px;
                    text-align: center;
                    margin-bottom: 10px;
                    width: 100%;
                    font-family: 'Helvetica', sans-serif;
                }

                .red-button {
                    height: 100px;
                    width: 100px;
                    border-radius: 50%;
                    background-color: red;
                    color: white;
                    border: none;
                    font-size: 16px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: background-color 0.3s;
                }

                .red-button:hover {
                    background-color: darkred;
                }

                .square-obj {
                    height: 100px;
                    width: 100px;
                    background-color: yellow;
                    margin: auto;
                }

                .circle-obj {
                    height: 100px;
                    width: 100px;
                    border-radius: 50%;
                    background-color: red;
                    margin: auto;
                }

                .star-obj {
                    background-repeat: no-repeat !important;
                    background-position: center !important;
                }

                .goal-display, .workspace {
                    height: 400px;  /* Fixed height instead of vh */
                    width: 30%;
                    min-width: 300px;
                    padding: 20px;
                    box-sizing: border-box;
                }

                .square-obj, .circle-obj {
                    background-repeat: repeat !important;
                    background-size: 20px 20px;
                    position: relative;
                    overflow: hidden;
                }
                
                .circle-obj {
                    border-radius: 50%;
                    overflow: hidden;
                }
                
                .star-obj {
                    background-repeat: no-repeat !important;
                    background-position: center !important;
                }
            `;
            document.head.appendChild(styles);

            // Initialize the goal from the passed data
            this.currentGoal = {
                a: { 
                    id: 'fixed-obj-a', 
                    shape: trial.goalState[0].shape === 'triangle' ? 'star' : trial.goalState[0].shape,  // Map triangle to star
                    color: trial.goalState[0].shade,
                    pattern: trial.goalState[0].texture === 'none' ? 'plain' : trial.goalState[0].texture 
                },
                b: { 
                    id: 'fixed-obj-b', 
                    shape: trial.goalState[1].shape === 'triangle' ? 'star' : trial.goalState[1].shape,  // Map triangle to star
                    color: trial.goalState[1].shade,
                    pattern: trial.goalState[1].texture === 'none' ? 'plain' : trial.goalState[1].texture 
                },
                c: { 
                    id: 'fixed-obj-c', 
                    shape: trial.goalState[2].shape === 'triangle' ? 'star' : trial.goalState[2].shape,  // Map triangle to star
                    color: trial.goalState[2].shade,
                    pattern: trial.goalState[2].texture === 'none' ? 'plain' : trial.goalState[2].texture 
                }
            };

            this.currentConfig = {
                a: { 
                    id: 'demo-obj-a', 
                    shape: 'star',
                    color: '1',
                    pattern: 'striped' 
                },
                b: { 
                    id: 'demo-obj-b', 
                    shape: 'square', 
                    color: '1',
                    pattern: 'plain' 
                },
                c: { 
                    id: 'demo-obj-c', 
                    shape: 'circle',
                    color: '2',
                    pattern: 'dotted' 
                }
            };

            // Display both goal and current state
            this.createGoalShapes(Object.values(this.currentGoal));
            console.log("Current Goal after creation:", this.currentGoal);
            this.displayShapes(this.currentConfig);

            // Add event listeners to shapes in current state
            this.addSelectListeners();

            // Add event listener to goal abandonment button
            document.getElementById('large-red-button').addEventListener('click', () => {
                this.abandonGoal();
            });
        }

        displayShapes(config) {
            // Function to display shapes based on the configuration
            for (const key in config) {
                const shapeData = config[key];
                const holderId = `demo-holder-${key.charAt(key.length - 1)}`;
                const shapeElement = this.makeShape(shapeData.id, shapeData.shape, shapeData.color, shapeData.pattern);
                document.getElementById(holderId).appendChild(shapeElement);
            }
        }

        makeShape(id, shape, color, pattern) {
            const shapeDiv = document.createElement('div');
            shapeDiv.id = id;
            
            const shapeClass = shape === 'star' ? 'star' : 
                              shape === 'circle' ? 'circle' : 'square';
            
            shapeDiv.className = `${shapeClass}-obj`;
            shapeDiv.style.backgroundColor = this.getShadeColor(color);
            shapeDiv.style.cursor = 'pointer';
            
            // Set base styles for all shapes
            shapeDiv.style.width = '100px';
            shapeDiv.style.height = '100px';
            shapeDiv.style.position = 'relative';
            
            // Apply patterns with better coverage
            if (pattern === 'striped' || pattern === 'stripe') {
                shapeDiv.style.backgroundImage = `repeating-linear-gradient(
                    45deg,
                    rgba(255, 255, 255, 0.8) 0px,
                    rgba(255, 255, 255, 0.8) 8px,
                    transparent 8px,
                    transparent 16px
                )`;
                shapeDiv.style.backgroundSize = '20px 20px';  // Smaller size for better coverage
                shapeDiv.style.backgroundRepeat = 'repeat';
            } else if (pattern === 'dotted') {
                shapeDiv.style.backgroundImage = `radial-gradient(
                    circle at 5px 5px,
                    rgba(255, 255, 255, 0.8) 2.5px,
                    transparent 2.5px
                )`;
                shapeDiv.style.backgroundSize = '12px 12px';  // Smaller size for better coverage
                shapeDiv.style.backgroundRepeat = 'repeat';
                shapeDiv.style.backgroundPosition = '0 0';    // Start from corner for better coverage
            }

            // Special handling for stars with updated pattern coverage
            if (shapeClass === 'star') {
                const canvas = document.createElement('canvas');
                canvas.width = 100;
                canvas.height = 100;
                const ctx = canvas.getContext('2d');
                
                // Draw star
                ctx.translate(50, 50);
                ctx.beginPath();
                
                const spikes = 5;
                const outerRadius = 40;
                const innerRadius = 20;
                
                for (let i = 0; i < spikes * 2; i++) {
                    const radius = i % 2 === 0 ? outerRadius : innerRadius;
                    const angle = (i * Math.PI) / spikes - Math.PI / 2;
                    if (i === 0) {
                        ctx.moveTo(radius * Math.cos(angle), radius * Math.sin(angle));
                    } else {
                        ctx.lineTo(radius * Math.cos(angle), radius * Math.sin(angle));
                    }
                }
                ctx.closePath();
                
                // Fill with base color
                ctx.fillStyle = this.getShadeColor(color);
                ctx.fill();
                
                // Apply pattern if needed with better coverage
                if (pattern === 'striped' || pattern === 'dotted') {
                    ctx.save();
                    ctx.clip();
                    
                    if (pattern === 'striped') {
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                        ctx.lineWidth = 4;
                        
                        for (let i = -80; i < 80; i += 8) {
                            ctx.beginPath();
                            ctx.moveTo(i - 40, -40);
                            ctx.lineTo(i + 40, 40);
                            ctx.stroke();
                        }
                    } else if (pattern === 'dotted') {
                        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                        // Increase dot coverage area and density
                        for (let x = -40; x <= 40; x += 6) {  // Wider range and smaller spacing
                            for (let y = -40; y <= 40; y += 6) {  // Wider range and smaller spacing
                                ctx.beginPath();
                                ctx.arc(x, y, 2, 0, Math.PI * 2);  // Slightly smaller dots
                                ctx.fill();
                            }
                        }
                    }
                    ctx.restore();
                }
                
                shapeDiv.style.backgroundImage = `url(${canvas.toDataURL()})`;
                shapeDiv.style.backgroundSize = 'contain';
                shapeDiv.style.backgroundColor = 'transparent';
            }
            
            return shapeDiv;
        }

        getShadeColor(shade) {
            return this.colorCode[shade] || "white";
        }

        addSelectListeners() {
            // Add click listeners to shapes in current state
            ['a', 'b', 'c'].forEach(key => {
                const shapeId = `demo-obj-${key}`;
                const shape = document.getElementById(shapeId);
                if (shape) {
                    shape.onclick = () => this.selectObj(shapeId, this.currentGoal);
                }
            });
        }

        selectObj(id, goal) {
            // console.log(`Selected: ${id}`, goal[id]);  // Comment out
            
            if (this.currentSelect.length === 0) {
                this.currentSelect.push(id);
                this.getEl(this.getHolderId(id)).style.border = 'solid gold 5px';
            } else if (this.currentSelect.length === 1) {
                if (this.currentSelect[0] === id) {
                    // Deselect if the same shape is clicked
                    this.currentSelect = [];
                    this.getEl(this.getHolderId(id)).style.border = '';
                } else {
                    // Select the second shape
                    this.currentSelect.push(id);
                    this.getEl(this.getHolderId(id)).style.border = 'dashed grey 5px';
                    setTimeout(() => {
                        this.makeTransition(this.currentSelect[0], this.currentSelect[1], goal);
                    }, 1000);
                }
            }
        }

        abandonGoal() {
            // console.log("Goal abandoned!");  // Comment out
            const abandonData = {
                // Trial metadata
                trial_type: 'goal_pursuit_action',
                trial_index: this.jsPsych.getProgress().current_trial_global,
                goal_trial_number: Math.floor(this.jsPsych.getProgress().current_trial_global / 2),
                time_elapsed: this.jsPsych.getTotalTime(),
                timestamp: new Date().toISOString(),
                participant_id: this.trialParams.participant_id,
                
                // Trial outcome
                success: false,
                abandoned: true,
                total_moves: this.currentSelect.length / 2,
                
                // States
                initial_state: this.initialConfig,
                final_state: this.currentConfig,
                goal_state: this.currentGoal,
                
                // Interaction data
                interactions: JSON.stringify(this.currentSelect),
                interaction_number: this.interactionCount,
                
                // Additional metadata
                completion_time: Date.now() - this.startTime,
                participant_id: this.jsPsych.getProgress().participant_id
            };
            
            console.log('Abandon Data:', JSON.stringify(abandonData, null, 2));
            this.jsPsych.finishTrial(abandonData);
        }

        // Additional functions from temp_pursuit.js
        safeColorChange(color, direction) {
            let colorInt = parseInt(color);
            if (direction === '+') {
                colorInt = colorInt + 1;
                if (colorInt > 5) colorInt = 1;
            }
            if (direction === '-') {
                colorInt = colorInt - 1;
                if (colorInt < 1) colorInt = 5;
            }
            return colorInt.toString();
        }

        createGoalShapes(fixedShapes) {
            fixedShapes.forEach(({ id, shape, color, pattern }) => {
                const shapeElement = this.makeShape(id, shape, color, pattern);
                this.getEl('fixed-holder-' + id.split('-')[2]).append(shapeElement);
            });
        }

        getHolderId(id) {
            let [taskId, _, objId] = id.split('-');
            return [taskId, 'holder', objId].join('-');
        }

        getEl(elementID) {
            return document.getElementById(elementID);
        }

        createCustomElement(id, className, type = 'div') {
            let element = (["svg", "polygon"].indexOf(type) < 0) ?
                document.createElement(type) :
                document.createElementNS("http://www.w3.org/2000/svg", type);
            if (className.length > 0) element.setAttribute("class", className);
            if (id.length > 0) element.setAttribute("id", id);
            return element;
        }

        createText(h = "h1", text = 'hello') {
            let element = document.createElement(h);
            let tx = document.createTextNode(text);
            element.append(tx);
            return element;
        }

        createBtn(btnId, text = "Button", className = "task-button", on = true) {
            let btn = this.createCustomElement("button", className, btnId);
            btn.disabled = !on;
            if (text.length > 0) btn.append(document.createTextNode(text));
            return btn;
        }

        makeTransition(a, r, currentGoal) {
            this.interactionCount++;
            
            // read agent properties
            let agent = a.split('-')[2];
            let agent_color = this.currentConfig[agent]['color'];
            let agent_shape = this.currentConfig[agent]['shape'];

            // read recipient properties
            let recipient = r.split('-')[2];
            let recipient_color = this.currentConfig[recipient]['color'];
            let recipient_shape = this.currentConfig[recipient]['shape'];
            let recipient_pattern = this.currentConfig[recipient]['pattern'];

            // Store state before transition
            const previousState = JSON.parse(JSON.stringify(this.currentConfig));

            // decide transitions
            let ret_color, ret_shape, ret_pattern;
            
            // Color transition logic (unchanged)
            if (parseInt(agent_color) > parseInt(recipient_color)) {
                ret_color = this.safeColorChange(recipient_color, '+');
            } else if (parseInt(agent_color) < parseInt(recipient_color)) {
                ret_color = this.safeColorChange(recipient_color, '-');
            } else {
                ret_color = recipient_color;
            }

            // Shape transition logic (unchanged)
            ret_shape = (Math.random() < 0.8) ? agent_shape : recipient_shape;

            // Pattern transition logic - always alternate
            if (recipient_pattern === 'plain') {
                ret_pattern = Math.random() < 0.5 ? 'striped' : 'dotted';
            } else {
                ret_pattern = 'plain';
            }

            // Update the shape and register changes
            this.getEl(this.getHolderId(r)).innerHTML = '';
            this.getEl(this.getHolderId(r)).append(this.makeShape(r, ret_shape, ret_color, ret_pattern));
            this.getEl(r).onclick = () => this.selectObj(r, this.currentGoal);

            // register changes
            this.currentConfig[recipient] = {
                id: r,
                shape: ret_shape,
                color: ret_color,
                pattern: ret_pattern
            };

            // check goal fulfillment
            let goal_fulfilled = this.isGoalFulfilled(currentGoal);
            
            // Save data for this interaction
            const interactionData = {
                // Trial metadata
                trial_type: 'goal_pursuit_action',
                trial_index: this.jsPsych.getProgress().current_trial_global,
                goal_trial_number: Math.floor(this.jsPsych.getProgress().current_trial_global / 2),
                time_elapsed: this.jsPsych.getTotalTime(),
                timestamp: new Date().toISOString(),
                participant_id: this.trialParams.participant_id,
                
                // Action details
                actor_id: agent,
                actor_shape: agent_shape,
                actor_color: agent_color,
                recipient_id: recipient,
                recipient_shape: recipient_shape,
                recipient_color: recipient_color,
                recipient_pattern: recipient_pattern,
                
                // Outcome
                resulting_shape: ret_shape,
                resulting_color: ret_color,
                resulting_pattern: ret_pattern,
                
                // States
                state_before: previousState,
                state_after: this.currentConfig,
                goal_state: this.currentGoal,
                
                // Trial status
                action_number: this.currentSelect.length / 2,
                interaction_number: this.interactionCount,
                goal_achieved: goal_fulfilled,
                abandoned: false,
                completion_time: Date.now() - this.startTime
            };

            // Log the data being saved
            console.log('Interaction Data:', JSON.stringify(interactionData, null, 2));
            
            this.jsPsych.data.write(interactionData);

            if (goal_fulfilled) {
                // console.log("Goal fulfilled!");  // Comment out
                this.clearSelectionBorders();
                alert("Congratulations! You've achieved the goal!");
                this.jsPsych.finishTrial();
                return;
            } else {
                // console.log("Goal not yet fulfilled");  // Comment out
                this.clearSelectionBorders();
                this.currentSelect = [];
            }
        }

        isGoalFulfilled(currentState) {
            // console.log('Checking goal fulfillment...');  // Comment out
            // console.log('Current Goal:', this.currentGoal);  // Comment out
            // console.log('Current State:', this.currentConfig);  // Comment out
            
            for (let key of ['a', 'b', 'c']) {
                if (this.currentGoal[key].shape !== this.currentConfig[key].shape ||
                    this.currentGoal[key].color !== this.currentConfig[key].color ||
                    this.currentGoal[key].pattern !== this.currentConfig[key].pattern) {
                    // console.log(`Mismatch found in shape ${key}:`);  // Comment out
                    // console.log(`Goal: ${JSON.stringify(this.currentGoal[key])}`);  // Comment out
                    // console.log(`Current: ${JSON.stringify(this.currentConfig[key])}`);  // Comment out
                    return false;
                }
            }
            return true;
        }

        randomInit() {
            let config = {
                'a': {},
                'b': {},
                'c': {},
            };
            config['a']['color'] = this.sampleFromList(Object.keys(colorCode));
            config['b']['color'] = this.sampleFromList(Object.keys(colorCode));
            config['c']['color'] = this.sampleFromList(Object.keys(colorCode));

            // Update shape options to use 'star' instead of 'triangle'
            config['a']['shape'] = this.sampleFromList(['square', 'circle', 'star']);
            config['b']['shape'] = this.sampleFromList(['square', 'circle', 'star']);
            config['c']['shape'] = this.sampleFromList(['square', 'circle', 'star']);

            config['a']['pattern'] = this.sampleFromList(['plain', 'stripe']);
            config['b']['pattern'] = this.sampleFromList(['plain', 'stripe']);
            config['c']['pattern'] = this.sampleFromList(['plain', 'stripe']);

            return config;
        }

        sampleFromList(arr, n = 1, replace = true) {
            if (n == 1) {
                return (arr[Math.floor(Math.random() * arr.length)]);
            } else {
                let sampled = [];
                for (let j = 0; j < n; j++) {
                    let randomIndex = Math.floor(Math.random() * arr.length);
                    sampled.push(arr[randomIndex]);
                    if (!replace) {
                        arr.splice(randomIndex, 1);
                    }
                }
                return sampled;
            }
        }

        // Add new helper method for clearing borders
        clearSelectionBorders() {
            // Safely clear borders by checking if elements exist
            for (let id of this.currentSelect) {
                const holder = this.getEl(this.getHolderId(id));
                if (holder) {
                    holder.style.border = '';
                }
            }
        }
    }

    GoalPursuitPlugin.info = info;

    return GoalPursuitPlugin;

})(jsPsychModule);
  