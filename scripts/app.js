
// *************************************************
// MAIN Program
// *************************************************



/* load data

	USE_KNOWN_DATA = true:
			donn�es d�j� splitt� venant de l'exemple en python, et qui donnent une accuracy de 97%
			permet de comparer afin de valider l'algorythme
*/
const USE_KNOWN_DATA = false;

if (USE_KNOWN_DATA) {

	$.ajax({
	  url: 'data/breast_cancer_data_split.json',
	  async: false,
	  dataType: 'json',
	  success: function (response) {
		split_data = response; // do stuff with response.
	  }
	});
	
	X_train = split_data.X_train;
	X_test = split_data.X_test;
	y_train = split_data.y_train;
	y_test = split_data.y_test;	
	
	
} else {
	// charger les donn�es et le splitter en train/test
	$.ajax({
	  url: 'data/breast_cancer_data.json',
	  async: false,
	  dataType: 'json',
	  success: function (response) {
		data = response; 
	  }
	});
	X = data.data;
	y = data.target;
	var split_data = train_test_split(X, y, 0.2);
	X_train = split_data[0];
	X_test = split_data[1];
	y_train = split_data[2];
	y_test = split_data[3];
	
}



//# Adaboost classification with n weak classifiers
// training
n = 15; //00;
clf = new Adaboost(n);
clf.fit(X_train, y_train);


// test
y_pred = clf.predict(X_test);
//console.log(y_test);
//console.log(y_pred);

acc = accuracy(y_test, y_pred);
pre = precision(y_test, y_pred);
rec = recall(y_test, y_pred);
fsc = f1score(y_test, y_pred);
console.log ("Accuracy:", acc, "precision:", pre, "recall:", rec, "f1score:", fsc, "clf", clf);

//console.log(JSON.stringify(clf.clfs));



	

//###########################
// exemple de s�rialisation 
// 		pour r�utilisation dans une application
//###########################

// obtenir les data des DecisionStump
// normalement, on enregistrerait les donn�es dans un fichier ou localStorage
var sd = clf.GetStumpsData();
//todo: localStorage.setItem("sd", JSON.stringify(sd)) 

// plus tard:
//		on relit les donn�es de stumps
//		on initialise le ABPredicator et 
//		on fait la pr�diction avec un array de vecteurs de features
// note: ABPredictor utilise DecisionStump 

var stumps_data = JSON.parse(sd);
// todo: var stumps_data = JSON.parse(localStorage.getItem("sd"))
var abp = new ABPredictor(stumps_data);

// ici, on r�utilise les donn�es test pr�c�dentes, juste pour v�rifier que ABPredictor fonctionne correctement
// normalement, on passerait un array de vecteurs de features qui sont encore inconnus
var y_pred = abp.predict(X_test);
acc = accuracy(y_test, y_pred);
pre = precision(y_test, y_pred);
rec = recall(y_test, y_pred);
fsc = f1score(y_test, y_pred);

// les r�sultats devraient �tre semblables � ce qu'on a obtenu pr�c�demment
console.log ("Accuracy:", acc, "precision:", pre, "recall:", rec, "f1score:", fsc, "clf", clf);

	
	