

//#########################
// # data processing
//#########################


const zip = (a, b) => a.map((k, i) => [k, b[i]]);


function range(n) {
	return Array.from(Array(n).keys());	
}

function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

function train_test_split(X, y, test_size=0.2) {
	indices = shuffle(range(X.length));
	split_idx = Math.floor(X.length * (1-test_size));
	
	var X_train = [],  X_test= [], y_train= [], y_test= [];
	for (var i = 0; i < split_idx; i++) {
		var arX = X[indices[i]];
		var arY = y[indices[i]];
		X_train.push(arX);
		y_train.push(arY);
	}
	for (var i = split_idx; i < X.length; i++) {
		var arX = X[indices[i]];
		var arY = y[indices[i]];
		X_test.push(arX);
		y_test.push(arY);		
	}
	return [X_train, X_test, y_train, y_test];
}




//#################################
// # mesures de performance
//#################################

function accuracy(y_true, y_pred) {
	var tptn = 0;
	for (let idx in y_true) {
		tptn += ( y_true[idx] == y_pred[idx] ? 1 : 0);
	}
    return  tptn / y_true.length;
}	
	
function precision(y_true, y_pred) {
	var tpfp = 0, tp = 0;
	for (let idx in y_true) {
		tp += ( y_true[idx] == 1 && y_pred[idx] == 1 ? 1 : 0);
		tpfp += y_pred[idx] == 1 ? 1 : 0;
	}
    return  tp / tpfp;
}
	
	
function recall(y_true, y_pred) {
	var tp = 0, tpfn = 0;
	for (let idx in y_true) {
		tp += ( y_true[idx] == 1 && y_pred[idx] == 1 ? 1 : 0);
		tpfn += y_true[idx] == 1 ? 1 : 0;
	}
    return  tp / tpfn;
}

function f1score(y_true, y_pred) {
	//F-Measure = (2 * Precision * Recall) / (Precision + Recall)
	p = precision(y_true, y_pred);
	r = recall(y_true, y_pred);
	return (2 * p * r) / (p + r);
}

