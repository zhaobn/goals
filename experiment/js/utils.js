
/* Custom wrappers */
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






/* Task specific **/
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
