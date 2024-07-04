def test_two():
    # Test case 1: Two of a feature anywhere
    state = State(shape1=Shape(sides=['square'], shade=['low'], texture=['present']),
                  shape2=Shape(sides=['circle'], shade=['medium'], texture=['not_present']),
                  shape3=Shape(sides=['triangle'], shade=['high'], texture=['present']))
    assert two('square', state) == True
    assert two('circle', state) == True
    assert two('triangle', state) == True
    assert two('low', state) == True
    assert two('medium', state) == True
    assert two('high', state) == True
    assert two('present', state) == True
    assert two('not_present', state) == True

    # Test case 2: Compare two shapes for given number of features
    state = State(shape1=Shape(sides=['square'], shade=['low'], texture=['present']),
                  shape2=Shape(sides=['circle'], shade=['medium'], texture=['not_present']),
                  shape3=Shape(sides=['triangle'], shade=['high'], texture=['present']))
    assert two('2', 'same', state) == False
    assert two('2', 'unique', state) == True
    assert two('3', 'same', state) == False
    assert two('3', 'unique', state) == False

    # Test case 3: Compare two shapes anywhere over all features
    state = State(shape1=Shape(sides=['square'], shade=['low'], texture=['present']),
                  shape2=Shape(sides=['circle'], shade=['medium'], texture=['not_present']),
                  shape3=Shape(sides=['triangle'], shade=['high'], texture=['present']))
    assert two('same', state) == False
    assert two('unique', state) == True

    # Test case 4: Compare two locations for one feature
    state = State(shape1=Shape(sides=['square'], shade=['low'], texture=['present']),
                  shape2=Shape(sides=['circle'], shade=['medium'], texture=['not_present']),
                  shape3=Shape(sides=['triangle'], shade=['high'], texture=['present']))
    assert two('square', '(0,1)', state) == True
    assert two('circle', '(0,1)', state) == False
    assert two('triangle', '(0,1)', state) == False
    assert two('low', '(0,1)', state) == True
    assert two('medium', '(0,1)', state) == False
    assert two('high', '(0,1)', state) == False
    assert two('present', '(0,1)', state) == True
    assert two('not_present', '(0,1)', state) == False

    # Test case 5: Compare two locations across all features
    state = State(shape1=Shape(sides=['square'], shade=['low'], texture=['present']),
                  shape2=Shape(sides=['circle'], shade=['medium'], texture=['not_present']),
                  shape3=Shape(sides=['triangle'], shade=['high'], texture=['present']))
    assert two('same', '0', state) == False
    assert two('same', '1', state) == False
    assert two('same', '2', state) == False

    # TODO: Add more test cases for comparing multiple features

    print("All tests passed!")

test_two()