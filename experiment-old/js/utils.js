

// Function to be executed when the button is clicked
function abandonGoal() {
  // Your code logic here
  console.log("Goal abandoned!"); // Example action
}

function safeColorChange(color, direction) {
  let colorInt = parseInt(color);

  if (direction=='+') {
    colorInt = colorInt + 1;
    (colorInt > 5) ? colorInt = 1 : null;

  }
  if (direction=='-') {
    colorInt = colorInt - 1;
    (colorInt < 0) ? colorInt = 5 : null;

  }

  return (colorInt.toString())
}

// Goal shapes do not change
function createGoalShapes(fixedShapes) {

  fixedShapes.forEach(({ id, shape, color, pattern }) => {
      console.log(`Color: ${color}`)
      const shapeElement = makeShape(id, shape, color, pattern);
      getEl('fixed-holder-' + id.split('-')[2]).append(shapeElement);
  });
}

function getHolderId(id) {
  let [taskId, _, objId] = id.split('-');
  return [taskId, 'holder', objId].join('-');
}

function getEl(elementID) {
  let el = document.getElementById(elementID)
  return el
}

function createCustomElement (id, className, type = 'div') {
  let element = (["svg", "polygon"].indexOf(type) < 0)?
    document.createElement(type):
    document.createElementNS("http://www.w3.org/2000/svg", type);
  if (className.length > 0) element.setAttribute("class", className);
  if (id.length > 0) element.setAttribute("id", id);
  return element;
}

function createText(h = "h1", text = 'hello') {
  let element = document.createElement(h);
  let tx = document.createTextNode(text);
  element.append(tx);
  return(element)
}

function createBtn (btnId, text = "Button", className = "task-button", on = true) {
  let btn = createCustomElement("button", className, btnId);
  btn.disabled = !on;
  (text.length > 0) ? btn.append(document.createTextNode(text)): null;
  return(btn)
}

function selectObj(id, currentGoal) {

  if (currentSelect.length == 0) {
    currentSelect.push(id);
    getEl(getHolderId(id)).style.border = 'solid gold 5px';


  } else if (currentSelect.length == 1) {

    if (currentSelect[0] == id) {
      currentSelect = [];
      getEl(getHolderId(id)).style.border = '';

    } else {
      currentSelect.push(id);
      getEl(getHolderId(id)).style.border = 'dashed grey 5px';
      setTimeout(() => {
        makeTransition(currentSelect[0], currentSelect[1], currentGoal);
      }, 1000);


    }

  }

}

function setAttributes(el, attrs) {
  for(var key in attrs) {
    el.setAttribute(key, attrs[key]);
  }
}

function sampleFromList(arr, n=1, replace=true) {
  if (n==1) {
    return (arr[Math.floor(Math.random()*arr.length)])
  }
  else {
    let sampled = [];
    for (let j = 0; j < n; j++) {
      let randomIndex = Math.floor(Math.random()*arr.length);
      sampled.push(arr[randomIndex]);
      if (replace == 0) {
        arr.splice(randomIndex, 1)[0];
      }
    }
    return sampled;
  }
}

const colorCode = {
  '1': '#ADD8E6',
  '2': '#87CEEB',
  '3': '#6495ED',
  '4': '#4682B4',
  '5': '#483D8B',
}

function makeShape(id, shape, color, pattern) {
  let className = shape + '-obj';
  let fillColor = colorCode[color]
  let div = createCustomElement(id, className);
  if (shape=='triangle') {
    div.style.borderBottom = `140px solid ${fillColor}`
  } else {
    div.style.backgroundColor = fillColor;
  }

  if (pattern=='stripe') {
    div.style.backgroundImage = `linear-gradient(45deg, #000 25%, transparent 25%, transparent 50%, #000 50%, #000 75%, transparent 75%, ${fillColor})`;
    div.style.backgroundSize = '50px 50px';

  }

  return div;
}

function makeTransition(a, r, currentGoal) {
  // read agent properties
  let agent = a.split('-')[2];
  let agent_color = currentConfig[agent]['color'];
  let agent_shape = currentConfig[agent]['shape'];

  // read recipient properties
  let recipient = r.split('-')[2];
  let recipient_color = currentConfig[recipient]['color'];
  let recipient_shape = currentConfig[recipient]['shape'];
  let recipient_pattern = currentConfig[recipient]['pattern'];

  // decide transitions
  let [ret_color, ret_shape, ret_pattern ] = [ '', '', '' ];


  if (parseInt(agent_color) > parseInt(recipient_color)) {
    ret_color = safeColorChange(recipient_color, '+');

  } else if (parseInt(agent_color) < parseInt(recipient_color)) {
    ret_color = safeColorChange(recipient_color, '-')

  } else {
    ret_color = recipient_color;
  }


  ret_shape = (Math.random() < 0.8) ? agent_shape : recipient_shape;

  ret_pattern = (recipient_pattern=='plain')? 'stripe' : 'plain';

  getEl(getHolderId(r)).innerHTML = '';
  getEl(getHolderId(r)).append(makeShape(r, ret_shape, ret_color, ret_pattern));
  getEl(r).onclick = () => selectObj(r);

  // register changes
  currentConfig[recipient]['color'] = ret_color;
  currentConfig[recipient]['shape'] = ret_shape;
  currentConfig[recipient]['pattern'] = ret_pattern;


  // check goal fullfillment
  let goal_fulfilled = false;
  goal_fulfilled = isGoalFulfilled(currentGoal);


  if (goal_fulfilled) {
    console.log("Goal fulfilled!");
    // You can add additional logic here, such as updating the UI or notifying the user
  } else {
    console.log("Goal not yet fulfilled.");
  }

  // clear up
  getEl(getHolderId(currentSelect[0])).style.border = '';
  getEl(getHolderId(currentSelect[1])).style.border = '';
  currentSelect = [];
}

function stateToString(state){
  let stateString = 
  `
  ${state['a'].color}
  ${state['a'].shape}
  ${state['a'].pattern}
  ${state['b'].color}
  ${state['b'].shape}
  ${state['b'].pattern}
  ${state['c'].color}
  ${state['c'].shape}
  ${state['c'].pattern}
  `;
  return stateString
}

// Function to check if the current configuration matches the current goal
function isGoalFulfilled(currentState) {

  console.log('Eval happening...');
  console.log(`Current Goal: ${currentGoal}`);
  let goal_fulfilled = false;

  goal_fulfilled = stateToString(currentGoal)==stateToString(currentConfig);

  return goal_fulfilled; // Return true if all properties match
}

function randomInit() {
  let config = {
    'a': {},
    'b': {},
    'c': {},
  };
  config['a']['color'] = sampleFromList(Object.keys(colorCode));
  config['b']['color'] = sampleFromList(Object.keys(colorCode));
  config['c']['color'] = sampleFromList(Object.keys(colorCode));

  config['a']['shape'] = sampleFromList(['square', 'circle', 'triangle']);
  config['b']['shape'] = sampleFromList(['square', 'circle', 'triangle']);
  config['c']['shape'] = sampleFromList(['square', 'circle', 'triangle']);

  config['a']['pattern'] = sampleFromList(['plain', 'stripe']);
  config['b']['pattern'] = sampleFromList(['plain', 'stripe']);
  config['c']['pattern'] = sampleFromList(['plain', 'stripe']);

  return config
}
