/**
	Adaboost
	
		inspiré de:  https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/adaboost.py
*/

/**

	DecisionStump

*/
function DecisionStump() {
	this.polarity = 1;
	this.feature_idx = null;
	this.threshold = null;
	this.alpha = null;
	this.min_error = Infinity; // information seulement
	
	this.predict = function (X) {
		var X_column = X.map(e => e[this.feature_idx]); 	
        var predictions;
		
        if (this.polarity == 1) {
			predictions = X_column.map(e => (e < this.threshold) ? -1 : 1);
        } else {
			predictions = X_column.map(e => (e > this.threshold) ? -1 : 1);
		}
        return predictions;
	}
}



/**

Adaboost

*/
function Adaboost(n_clf=5) {
	this.n_clf = n_clf;
	
	this.fit = function (X, y) {
        
		var n_samples = X.length;
		var n_features = X[0].length;
		
        // Initialize weights to 1/N
		var w = new Array(n_samples).fill(1 / n_samples);
        this.clfs = [];
		
		// Iterate through classifiers
        for (_ in range(this.n_clf) ) {
		
            var clf = new DecisionStump();
			var min_error = Infinity; 
			
			// greedy search to find best threshold and feature
            for (var feature_i in range(n_features) ) {

                var X_column = X.map(e => e[feature_i]); 
                var thresholds = Array.from(new Set(X_column));// unique

                for (var idx in thresholds) {
					var threshold = thresholds[idx];
                    //# predict with polarity 1
                    var p = 1;
                   
					var predictions = X_column.map(e => (e < threshold) ? -1 : 1);
					
                    //# Error = sum of weights of misclassified samples;
					var error = 0;
					for (var i=0; i< y.length; i++) {
						error += ( predictions[i] != y[i] ) ? w[i] : 0;
					}
					
					// reverse 
					if (error > 0.5) {
						error = 1 - error;
						p = -1;
					}										

                    //# store the best configuration
                    if (error < min_error) {					
						min_error = error;
                        clf.polarity = p;
                        clf.threshold = threshold;
                        clf.feature_idx = feature_i;
                    }
				}				
			}
			
			
			//# calculate alpha
			var EPS = 1e-10;
			clf.alpha = 0.5 * Math.log((1.0 - min_error + EPS) / (min_error + EPS));

			// ajouté jp
			clf.min_error = min_error;
			
			//## calculate predictions and update weights
			//# Normalize to one
			var predictions = clf.predict(X);			
			// # w *= np.exp(-clf.alpha * y * predictions)
			var div = 0;
			for (var i = 0; i < y.length; i++) {
				w[i] *= Math.exp(-clf.alpha * y[i] * predictions[i]);
				div += w[i];
			}
			w = w.map(e => e / div);
			
			
			//# Save classifier
			this.clfs.push(clf);
			  
			 // test early stopping (jp)
			//if (clf.min_error > clf.alpha ) {
			if ( clf.alpha < 0.3) {
				return;
			}
		
		}
	} // end fit
	
	
	
	
	// pour faire la validation des données test
	this.predict = function (X) {
      
		// prédiction pour chaque stump
        var clf_preds = this.clfs.map(clf => clf.predict(X).map(e => e * clf.alpha) );

		// array pour prédictions sur chaque sample sommées par feature		
		// faire les sommes par colonne (feature de chaque clf pour avoir un vecteur de prédiction)
		var y_pred = new Array(clf_preds[0].length).fill(0);
		for (var idx in clf_preds) {			
			for (var j in clf_preds[idx]) {
				y_pred[j] += clf_preds[idx][j];
			}
		}
		y_pred = y_pred.map(e => Math.sign(e));		
        return y_pred;
	}	
	
	
	// export les valeurs de DecisonStrumps (clfs)  en json
	this.GetStumpsData = function() {
		var j = JSON.stringify(this.clfs);	
		return j;
	}
}





/**

ABPredictor: 
	fait la prédiction de X à l'aide des stumps définis par clfs_data
	
		clfs_data: 
			array [
				{"polarity":-1,"feature_idx":"23","threshold":787.9,"alpha":0.9509461935349698,"min_error":0.12989444331061317},
				...
			]
			
	Utilise DecisionStump
		
*/
function ABPredictor(clfs_data) {
	this.clfs = [];
	for (idx in clfs_data) {
		var ds = new DecisionStump();
		ds.polarity = clfs_data[idx].polarity;
		ds.feature_idx = clfs_data[idx].feature_idx;
		ds.threshold = clfs_data[idx].threshold;
		ds.alpha = clfs_data[idx].alpha;
		this.clfs.push(ds);
	}
	
	
	// faire la prédiction de X (X est un array de features)	
	this.predict = function (X) {
       
		// prédiction pour chaque stump
        var clf_preds = this.clfs.map(clf => clf.predict(X).map(e => e * clf.alpha) );

		// array pour prédictions sur chaque sample sommées par feature		
		// faire les sommes par colonnes (feature de chaque clf pour avoir un vecteur de prédiction)
		var y_pred = new Array(clf_preds[0].length).fill(0);
		for (var idx in clf_preds) {			
			for (var j in clf_preds[idx]) {
				y_pred[j] += clf_preds[idx][j];
			}
		}
		y_pred = y_pred.map(e => Math.sign(e));		
        return y_pred;
	}	
}

