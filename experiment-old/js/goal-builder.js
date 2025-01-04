let selectedShapes = [];
const maxSelections = 3;

function initializeGoalBuilder() {
    // Create the grid of available shapes
    const grid = createCustomElement('stimulus-grid', 'stimulus-grid');
    
    // Create all possible combinations
    const shapes = ['square', 'circle', 'triangle'];
    const colors = Object.keys(colorCode);
    const patterns = ['plain', 'stripe'];
    
    // Create preview area
    const previewContainer = createCustomElement('preview-container', 'preview-container');
    for (let i = 0; i < 3; i++) {
        const slot = createCustomElement(`preview-slot-${i}`, 'preview-slot');
        previewContainer.appendChild(slot);
    }
    
    // Add shapes to grid
    shapes.forEach(shape => {
        colors.forEach(color => {
            patterns.forEach(pattern => {
                const item = createCustomElement('', 'stimulus-item');
                const shapeEl = makeShape('', shape, color, pattern);
                
                item.onclick = () => selectShape({ shape, color, pattern });
                item.appendChild(shapeEl);
                grid.appendChild(item);
            });
        });
    });

    // Add elements to page
    document.body.appendChild(previewContainer);
    document.body.appendChild(grid);
}

function selectShape(shapeConfig) {
    if (selectedShapes.length >= maxSelections) {
        return;
    }
    
    // Add shape to selected array
    selectedShapes.push({
        id: `fixed-obj-${String.fromCharCode(97 + selectedShapes.length)}`, // generates 'a', 'b', 'c'
        ...shapeConfig
    });
    
    // Update preview
    updatePreview();
    
    // If we have all three shapes, create the goal
    if (selectedShapes.length === maxSelections) {
        createGoal();
    }
}

function updatePreview() {
    selectedShapes.forEach((shape, index) => {
        const slot = getEl(`preview-slot-${index}`);
        slot.innerHTML = '';
        slot.appendChild(makeShape('', shape.shape, shape.color, shape.pattern));
    });
}

function createGoal() {
    // Convert selected shapes into the goal format
    const goal = {};
    selectedShapes.forEach((shape, index) => {
        const key = String.fromCharCode(97 + index); // generates 'a', 'b', 'c'
        goal[key] = shape;
    });
    
    // You can now use this goal object as your currentGoal
    console.log('Created goal:', goal);
    return goal;
}



// Initialize when the page loads
window.addEventListener('load', initializeGoalBuilder);