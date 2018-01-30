tests_conf = {
        'Test for the Secom data set (linear).' : {
            'get_nodes': True,
            'type': 'linear',
            'name': 'secom',
            'data_info': {
                'file': ['src/tests/traindatas/secom/secom.data', 'src/tests/traindatas/secom/secom_labels.data'],
                'delimiter': ' ',
                'class_split': True,
                'header': None
                }
            },
        'Test for the credit card risk (linear).' : {
            'get_nodes': True,
            'type': 'linear',
            'name': 'credit card',
            'data_info': {
                'file': ['src/tests/traindatas/creditcard/credit_card.csv'],
                'delimiter': ',',
                'class_split': False,
                'index': 0,
                'label':{
                    -1: 0
                    }
                }
            },
        'Test for the artificial data set (linear).' : {
            'get_nodes': True,
            'type': 'linear',
            'name': 'artificial',
            'data_info': {
                'file': 'artificial linear',
                'delimiter': None,
                'class_split': None,
                'auto_gen': True
                }
            },
        'Test for the cancer data set (nonlinear).' : {
            'get_nodes': True,
            'type': 'nonlinear',
            'name': 'cancer',
            'data_info': {
                'file': 'cancer',
                'delimiter': None,
                'class_split': None,
                'auto_gen': True,
                'label':{
                    -1: 0
                    }
                }
            },
        'Test for the chess data set (nonlinear).' : {
            'get_nodes': False,
            'type': 'nonlinear',
            'name': 'artificial',
            'data_info': {
                'file': ['src/tests/traindatas/chess/chess_data.csv', 'src/tests/traindatas/chess/chess_class.csv'],
                'delimiter': ',',
                'class_split': True,
                }
            }
        }
