
let currentSelect = [];
let data = {};
randomInit()

let initialConfig = randomInit();
// { 'a': {'color': '1', 'shape': 'square', 'pattern': 'plain'},
//   'b': {'color': '5', 'shape': 'circle', 'pattern': 'stripe'},
//   'c': {'color': '3', 'shape': 'triangle', 'pattern': 'plain'} };
let currentConfig = Object.create(initialConfig);


getEl('demo-holder-a').append(makeShape('demo-obj-a', initialConfig['a']['shape'], initialConfig['a']['color'], initialConfig['a']['pattern']));
getEl('demo-holder-b').append(makeShape('demo-obj-b', initialConfig['b']['shape'], initialConfig['b']['color'], initialConfig['b']['pattern']));
getEl('demo-holder-c').append(makeShape('demo-obj-c', initialConfig['c']['shape'], initialConfig['c']['color'], initialConfig['c']['pattern']));

getEl('demo-obj-a').onclick = () => selectObj('demo-obj-a');
getEl('demo-obj-b').onclick = () => selectObj('demo-obj-b');
getEl('demo-obj-c').onclick = () => selectObj('demo-obj-c');

function getHolderId(id) {
  let [taskId, _, objId] = id.split('-');
  return [taskId, 'holder', objId].join('-');
}

function selectObj(id) {

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
        makeTransition(currentSelect[0], currentSelect[1]);
      }, 1000);


    }

  }

}


function makeTransition(a, r) {
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



  // clear up
  getEl(getHolderId(currentSelect[0])).style.border = '';
  getEl(getHolderId(currentSelect[1])).style.border = '';
  currentSelect = [];


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
