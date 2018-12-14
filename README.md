Sentiment Analysis on News Articles using Machine Learning and Deep Learning Models
====================

##Description

   Η παρακάτω υλοποίηση εξετάζει το πρόβλημα της ανάλυσης συναισθήματος (sentiment analysis), δηλαδή της αυτόματη κατηγοριοποίηση ενός κειμένου ανάμεσα σε θετικό, αρνητικό ή ουδέτερο, με βάση την άποψη του συγγραφέα πάνω στο θέμα που αναλύεται σε αυτό. Τα δεδομένα εκπαίδευσης των μοντέλων που χρησιμοποιήθηκαν αποτέλεσαν ειδησεογραφικά άρθρα ελληνικού περιεχομένου, ενώ εξήχθησαν μέσω του News API και μέσω δύο κριτών αξιολογήθηκαν ώς πρός το συναίσθημα με βάση τα περιεχόμενα τους όπου η τομή της συμφωνίας των άρθρων δημιούργησε το αρχικό σετ δεδομένων. Στην συνέχεια, το αρχικό σετ δεδομένων υπέστει μια σειρά βημάτων προεπεξεργασίας, καθώς χρησιμοποιήθηκε επιπλέον η χρήση αυτόματης περίληψης, αφαίρεση των άρθρων με ουδέτερη κατηγορία συναισθήματος και χρήση POS (part of speech) Tagger με αποτέλεσμα την δημιουργία πέντε συνολικά διαφορετικών σετ δεδομένων. Έπειτα, εφαρμόστηκαν τόσο ένα σύνολο απο τεχνικές μηχανικής μάθησης όσο και τεχνικές Νευρωνικών Δικτύων, με σκοπό την αναγνώριση του συναισθήματος των ειδησεογραφικών άρθρων και την κατηγοριοποίηση τους. Τέλος, για την αξιολόγηση των μοντέλων χρησιμοποιήθηκε  η μέθοδος του Cross Validation.

##Libraries Used

   Για την υλοποίηση των μοντέλων Μηχανικής Μάθησης και Βαθιών Νευρωνικών Δικτύων χρησιμοποιήθηκαν κατα κύριο λόγο οι βιβλιοθήκες Keras, Scikit Learn, Pandas, Numpy, Tensorflow σε συνδυασμό με βιβλιοθήκες για την υλοποίηση δευτερευόντων λειτουργιών.

##Datasets and Descriptions

   Τα αρχεία που χρησιμοποιήθηκαν βρίσκονται στον φάκελο Datasets Used και βρίσκονται το κάθε ένα στους επιμέρους φακέλους Machine Learning και Deep Learning όπου στο φάκελο Deep Learning βρίσκονται με την αλλαγή στην κατάληξη του ονόματος να τελιώνει σε 2 (π.χ. Nopreprocess2.csv) και αναφέρονται παρακάτω συνοδευόμενα απο τις περιγραφές τους:

1. POS_Tagged.csv : Σετ δεδομένων με τις 3 κατηγορίες συναισθήματος όπου έχει γίνει χρήση προεπεξεργασίας των δεδομένων και χρήση POS Tagger.

2. with_neutr_summ_25.csv : Σετ δεδομένων με τις 3 κατηγορίες συναισθήματος όπου έχει γίνει προεπεξεργασία των δεδομένων και χρήση αυτόματης περίληψης με τον αλγόριθμο TextRank.

3. Nopreprocess.csv :  Σετ δεδομένων με τις 3 κατηγορίες συναισθήματος χωρίς να γίνει κάποια προεπεξεργασία του κειμένου.

4. Preprocessed.csv : Σετ δεδομένων με τις 3 κατηγορίες συναισθήματος όπου έχουν γίνει τα βήματα προεπεξεργασίας που αναλύθηκαν στο έγγραφο της Α.Δ.Ε.

5. Preprocessed_Without_Neutral.csv : Σετ δεδομένων χωρίς την ουδέτερη κατηγορία συναισθήματος όπου έχουν γίνει τα βήματα προεπεξεργασίας που αναλύθηκαν στο έγγραφο της Α.Δ.Ε.

   Επιπλέον στον φάκελο με το όνομα Not_used_Datasets, υπάρχουν τα σετ δεδομένων που δεν χρησιμοποιήθηκαν για την εξαγωγή των αποτελεσμάτων και είναι τα παρακάτω:

1. With_neutr_summ_25luhn.csv : Σετ δεδομένων με τις 3 κατηγορίες συναισθήματος όπου έχει γίνει προεπεξεργασία των δεδομένων και χρήση αυτόματης περίληψης με τον αλγόριθμο του Luhn. 

2. With_neutr_summ_25lsa.csv : Σετ δεδομένων με τις 3 κατηγορίες συναισθήματος όπου έχει γίνει προεπεξεργασία των δεδομένων και χρήση αυτόματης περίληψης με LSA.

##Machine Learning Models

   Στον φάκελο με το όνομα Machine Learning υπάρχουν τα μοντέλα μηχανικής μάθησης που υλοποιήθηκαν καθώς και η υλοποίηση για την επιλογή των παραμέτρων του κάθε μοντέλου. Συγκεκριμένα:

* LR.py : Υλοποίηση μοντέλου λογιστικής παλινδρόμησης και εξαγωγή αποτελεσμάτων μέσω Cross Validation.
* NB.py : Υλοποίηση μοντέλου Naive Bayes και εξαγωγή αποτελεσμάτων μέσω Cross Validation.
* RF.py : Υλοποίηση μοντέλου Random Forest και εξαγωγή αποτελεσμάτων μέσω Cross Validation.
* SGD.py : Υλοποίηση μοντέλου Linear SVM με SGD και εξαγωγή αποτελεσμάτων μέσω Cross Validation.
* Train-TestSplit-Models.py : Υλοποίηση όλων των μοντέλων για όλα χαρακτηριστικά και εξαγωγή αποτελεσμάτων με τον χωρισμό των δεδομένων ώς 80% δεδομένα εκπαίδευσης και 20% δεδομένα ελέγχου.
* tunerLR.py : Υλοποίηση GridSearch με Cross Validation για την επιλογή των παραμέτρων του μοντέλου Logistic Regression.
* tunerNB.py : Υλοποίηση GridSearch με Cross Validation για την επιλογή των παραμέτρων του μοντέλου Naive Bayes.
* tunerSGD.py : Υλοποίηση GridSearch με Cross Validation για την επιλογή των παραμέτρων του μοντέλου Linear SVM με SGD.

##Deep Learning Models

* LSTM2.py : Υλοποίηση μοντέλου LSTM (Long Short Term Memory) δικτύου που δέχεται σαν είσοδο όλα τα σετ δεδομένων που υπάρχουν 2 κατηγορίες συναισθήματος, δηλαδή αυτά που έχει αφαιρεθεί η ουδέτερη κατηγορία.
* LSTM-CNN2.py : Υλοποίηση συνδυασμού μοντέλων LSTM (Long Short Term Memory) και CNN (Convolutional Neural Network) που δέχεται σαν είσοδο όλα τα σετ δεδομένων που υπάρχουν 2 κατηγορίες συναισθήματος, δηλαδή αυτά που έχει αφαιρεθεί η ουδέτερη κατηγορία.
* LSTM3.py : Υλοποίηση μοντέλου LSTM (Long Short Term Memory) δικτύου που δέχεται σαν είσοδο όλα τα σετ δεδομένων που υπάρχουν 3 κατηγορίες συναισθήματος.
* LSTM-CNN3.py : Υλοποίηση συνδυασμού μοντέλων LSTM (Long Short Term Memory) και CNN (Convolutional Neural Network) που δέχεται σαν είσοδο όλα τα σετ δεδομένων που υπάρχουν 3 κατηγορίες συναισθήματος.

#Data Preprocessing

   Στον φάκελο Preprocessing υπάρχουν όλα τα αρχεία που χρησιμοποιήθηκαν για την προεπεξεργασία των δεδομένων και την δημιουργία όλων των σετ δεδομένων.

#How to Run





```bash
The Machine Learning directory contains the final source code of our research
regarding Machine Learning Algorithms along with a simple User Interface
to allow experimentation with a variety of algorithms.
```

## Feature Analysis

```bash
Feature Analysis is a project that analyses raw data exported from our research
experiments and produces the appropriate output which is later used to
train our classifiers. 

It produces the appropriate output according to the following input:

  * Days for recent/other separation
  * Training Window
  * Offset
  * Labeling Window
  * Labeling Percentage
  * Youtube binary
  * Twitter Binary
```
## Doc

```bash
Detailed documentation about the part of our research that is related
to Machine Learning Experimentation
```

## Testing

```bash
Any changes to the source code are first published here to fix any bugs, 
undergo testing and ensure that the results of the machine learning algorithms 
are 100% valid before they are moved to the Research Interface.
```

