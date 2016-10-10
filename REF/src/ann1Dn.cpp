
// ann1d.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "trace.h"

#include "lib\signal.h"
#include "libnn\network.h"
#include "libnn\neuron.h"



typedef struct _acur {
        float se;
        float sp;
        float pp;
        float np;
        float ac;
} ACUR, *PACUR;


enum NORMALIZATION {NONE, MINMAX, ZSCORE, SIGMOIDAL, ENERGY};
wchar_t normalization_type[][20] = {L"minmax", L"zscore", L"sigmoidal", L"energy"};


int normalization = 0;        
int validation_type = 0;
int vector_length = 0;     //size of the first vector read from read_class() trn,vld or tst set

class ANNetwork *ann = 0; 


vector<CSignal *> signals;     //file mapping classes of signals
//if used energy or minmax for a vector, filemapping is overwritten with new data





void read_class(FILE *fp, PREC rec, int c = 0);   //closes fp handle
int read_line(FILE *f, wchar_t *buff, int *c = 0);
void get_file_name(wchar_t *path, wchar_t *name);
int parse_path(wchar_t *path, wchar_t *dir, wchar_t *name);
void msec_to_time(int msec, int& h, int& m, int& s, int& ms);


void train(int argc, wchar_t* argv[]);
void validate(PREC rec, float TH, float *acur, PACUR pacur);
float gmean(float m, int n);    //geometric mean pow(m,1/n);
void set_validation(PREC vld, PREC trn, float p);
void dump_sets(PREC trn, PREC vld, PREC tst);
void test(int argc, wchar_t* argv[]);
void set_normalization(REC *rec, ANNetwork *pann);             //save normalization params to ANN input layer




/*
  //training mode
   1 t          //train
   2 cnf.nn     //network conf file
   3 cls1.txt   //files for class1  [0.9]
   4 cls2.txt   //files for class2  [0.1]
   5 epochs     //epochs num
    6 [val.txt]  //validation set
    7 [tst.txt]  //test set
    8 [val TH]   //validation threshold decision
    9 [val type] //validation type (mse/ac/...)
   10 [norm]     //normilize input data [0-none],1-minmax,2-zscore,3-softmax,4-energy,5-minimaxscale
   11 [error]    //default 0.05

  //test mode
   1 r          //run
   2 cnf.nn     //network file
   3 cls.txt    //files to test
   4 [TH 0.5]   //threshold optional only for 4-energy norm
   5 [norm]     //normilize input data [0-none],1-minmax,2-zscore,3-softmax,4-energy

                                                                                              */
int _tmain(int argc, wchar_t* argv[])
{
        srand((unsigned int)time(0));

        if (argc == 1) {
                wprintf(L"\n argv[1] t-train\n");
                wprintf(L" argv[2] network conf file\n");
                wprintf(L" argv[3] cls1 files [0.9]\n");
                wprintf(L" argv[4] cls2 files [0.1]\n");
                wprintf(L" argv[5] epochs num\n");
                wprintf(L"  argv[6] [validation class]\n");
                wprintf(L"  argv[7] [test class]\n");
                wprintf(L"  argv[8] [validation TH 0.5]\n");
                wprintf(L"  argv[9] [vld metric mse]\n");
                wprintf(L" argv[10] [norm]: [0-no], 1-minmax, 2-zscore, 3-softmax, 4-energy\n");
                wprintf(L" argv[11] [error tolerance cls] +- 0.05 default\n\n");

                wprintf(L" argv[1] r-run\n");
                wprintf(L" argv[2] network conf file\n");
                wprintf(L" argv[3] cls files\n");
                wprintf(L" argv[4] [validation TH 0.5]\n");
                wprintf(L" argv[5] [norm]: [0-no], 1-minmax, 2-zscore, 3-softmax, 4-energy\n\n");

                wprintf(L"  ann1dn.exe t net.nn cls1 cls2 3000 [tst.txt][val.txt][TH [0.5]][val type [mse]] [norm [0]] [err [0.05]] \n");
                wprintf(L"  ann1dn.exe r net.nn testcls [TH [0.5]] [norm [0]]\n\n");

                wprintf(L" metrics: [0 - mse]\n");
                wprintf(L"           1 - AC\n");
                wprintf(L"           2 - sqrt(SE*SP)\n");
                wprintf(L"           3 - sqrt(SE*PP)\n");
                wprintf(L"           4 - sqrt(SE*SP*AC)\n");
                wprintf(L"           5 - sqrt(SE*SP*PP*NP*AC)\n");
                wprintf(L"           6 - F-measure b=1\n");
                wprintf(L"           7 - F-measure b=1.5\n");
                wprintf(L"           8 - F-measure b=3\n");
        } else if (!wcscmp(argv[1], L"t"))
                train(argc, argv);
        else if (!wcscmp(argv[1], L"r"))
                test(argc, argv);
        else
                wprintf(L"argv[1] t-train, r-run\n");

        return 0;
}




///////////////////////////TRAINING/////////////////////////////////////////////////////////////
void train(int argc, wchar_t* argv[])
{
        //REC trncopy;          //for overall classification accuracy

        REC trnrec;           //class 1,2 records
        REC vldrec;           //validation records
        REC tstrec;           //test records
        bool vld = false;     //is there validation file?
        bool tst = false;     //is there test file


////parse optional arguments 6,7,8,9////////////////////////////////
        float TH = 0.5f;
        float error = 0.05f;
        if (argc >= 6 + 1) {
                if (wcslen(argv[6]) > 1) { // 6,7,8,9  test,validation class; TH; validation_type
                        if (argc >= 10 + 1)
                                normalization = _wtoi(argv[10]);
                        if (argc >= 11 + 1)
                                error = float(_wtof(argv[11]));

                        TH = float(_wtof(argv[8]));
                        validation_type = _wtoi(argv[9]);


                        //check validation set
                        FILE *val = _wfopen(argv[6], L"rt");
                        if (val) {
                                read_class(val, &vldrec);
                                if (vldrec.entries.size())
                                        wprintf(L" validation size: %d files, TH = %.2f\n", vldrec.entries.size(), TH);
                                else
                                        vld = true;
                        } else {
                                wprintf(L" failed to open %s\n", argv[6]);
                                exit(1);
                        }

                        //check test set
                        FILE *test = _wfopen(argv[7], L"rt");
                        if (test) {
                                read_class(test, &tstrec);
                                if (tstrec.entries.size())
                                        wprintf(L" test size: %d files\n", tstrec.entries.size());
                                else
                                        tst = true;
                        } else {
                                wprintf(L" failed to open %s\n", argv[7]);
                                exit(1);
                        }
                } else {
                        normalization = _wtoi(argv[6]);
                        if (argc >= 7 + 1)
                                error = float(_wtof(argv[7]));
                }
        }
////////////////////////////////////////////////////////////////////




        wprintf(L"loading data...\n");
        FILE *cls1 = _wfopen(argv[3], L"rt");
        FILE *cls2 = _wfopen(argv[4], L"rt");

        if (!cls1 || !cls2) {
                wprintf(L"failed to open files %s %s\n", argv[3], argv[4]);
                exit(1);
        } else {
                read_class(cls1, &trnrec, 1);   //by default put 1 class mark
                read_class(cls2, &trnrec, 2);   //by default put 2 class mark
        }

        if (!trnrec.entries.size()) {
                wprintf(L"no files loaded to training set.\n");
                exit(1);
        } else if (trnrec.clsnum.size() != 2) {
                wprintf(L"%d classes loaded. works only for 2 classes.\n", trnrec.clsnum.size());
                exit(1);
        } else
                wprintf(L" cls%d: %d  cls%d: %d  files loaded.  size: %d samples\n", trnrec.clsnum[0], trnrec.indices[0].size(), trnrec.clsnum[1], trnrec.indices[1].size(), trnrec.entries[0]->size);


        //arrange 25% from train set to validation/test sets
        if (vld && tst) {
                set_validation(&vldrec, &trnrec, 25.0f);
                set_validation(&tstrec, &trnrec, 35.0f);
        } else if (vld && !tst)
                set_validation(&vldrec, &trnrec, 50.0f);
        else if (!vld && tst)
                set_validation(&tstrec, &trnrec, 50.0f);

        dump_sets(&trnrec, &vldrec, &tstrec);



        //load network
        ann = new ANNetwork(argv[2]);
        if (ann->status() < 0) {
                wprintf(L"failed to load network: %s", argv[2]);
                exit(1);
        }
        if (ann->get_layer(0)->get_neurons_number() != vector_length) {
                if (ann->get_layer(0)->get_neurons_number() > vector_length) {
                        wprintf(L" input layer neurons %d are more than data dimension %d", ann->get_layer(0)->get_neurons_number(), vector_length);
                        exit(1);
                } else
                        wprintf(L" input layer neurons %d are less than data dimension %d\n", ann->get_layer(0)->get_neurons_number(), vector_length);
        }


        if (normalization && normalization != 4) { //energy normalization per vector
                wprintf(L"normalizing %s...\n", normalization_type[normalization-1]);
                set_normalization(&trnrec, ann);   //get normalization params  add,mult to ANN  from training set
        }



        int msecs = GetTickCount();
        wprintf(L"training...\n");

        float dvec[1] = {0.0f};
        float *ivec;
        float ovec[1] = {0.0f};
        float ovec1[1] = {0.0f}, ovec2[1] = {0.0f};
        bool prv = false;
        int x = 0, y = 0, ii = 0;
        int quit = 0;

        float acur = 0.0f, tmpacur;
        ACUR pacur, tmppacur;
        memset(&pacur, 0, sizeof(ACUR));
        memset(&tmppacur, 0, sizeof(ACUR));

        CTRACE trc(L"macurtrace.txt");
        ////////////////////TRAINING////////////////////////////////////////////////////////////////
        int step = (trnrec.indices[0].size() > trnrec.indices[1].size()) ? 2 * (int)trnrec.indices[0].size() : 2 * (int)trnrec.indices[1].size();
        int EPOCHS = _wtoi(argv[5]);
        int e = EPOCHS * step;
        int maxepoch = 0;
        while (e) {
                if (x > 1) {
                        x = 0;
                        ii++;
                }
                if (x == 0) //1st class
                        y = ii % (int)trnrec.indices[x].size();
                else if (x == 1) //2nd class
                        y = ii % (int)trnrec.indices[x].size();


                int ind = trnrec.indices[x].at(y);
                ivec = trnrec.entries[ind]->vec;

                int cls = trnrec.entries[ind]->cls;
                if (cls == 1) dvec[0] = 0.9f;
                else if (cls == 2) dvec[0] = 0.1f;

                ann->train(ivec, ovec, dvec, error);

                if (cls == 1) ovec1[0] += ovec[0];
                else if (cls == 2) ovec2[0] += ovec[0];

                x++;
                e--;

                ////////////////////////////////////////////////////////////////////////////////////////////////
                if (!(e % step)) { //one epochs is expired
                        float mout1 = ovec1[0] / (float(step) / 2.0f);   //mean out1
                        float mout2 = ovec2[0] / (float(step) / 2.0f);   //mean out2
                        ovec1[0] = 0.0f;
                        ovec2[0] = 0.0f;

                        if (quit == 10)  //no more error
                                break;

                        if (fabsl(mout1 - 0.9f) > error || fabsl(mout2 - 0.1f) > error)
                                quit = 0;
                        else
                                quit++;


                        //ann->save(L"temp.nn");
                        wprintf(L"  epoch: %d   out: %f %f ", EPOCHS - (e / step), mout1, mout2);

                        ////validate/////////////////////////////////////////////////////////////////////////////////////////////
                        if (vldrec.entries.size() && (mout1 > TH && mout2 < TH)) {
                                validate(&vldrec, TH, &tmpacur, &tmppacur);

                                trc.dump(tmpacur);
                                if (tmpacur >= acur) {
                                        maxepoch = EPOCHS - (e / step);
                                        acur = tmpacur;
                                        memcpy(&pacur, &tmppacur, sizeof(ACUR));
                                        if (!ann->save(L"maxacur.nn"))
                                                wprintf(L"  failed to save maxacur.nn  ");
                                }

                                wprintf(L"  max acur: %.2f (epoch %d)   se:%.2f sp:%.2f ac:%.2f\n", acur, maxepoch, pacur.se, pacur.sp, pacur.ac);
                        } else
                                wprintf(L"\n");
                        //////////////////////////////////////////////////////////////////////////////////////////////////////////


                        for (int i = 0; i < (int)trnrec.indices.size(); i++)  //shuffle indices to entries array
                                random_shuffle(trnrec.indices[i].begin(), trnrec.indices[i].end());


                        if (kbhit() && _getwch() == 'q') //quit program ?
                                e = 0;
                }//one epoch is expired/////////////////////////////////////////////////////////


        }
        ////////while(epochs)/////////////////////////////////////////////////////////////////////////////////

        if (e)
                wprintf(L"training done.\n");

        int hour, min, sec, msec;
        msec_to_time(GetTickCount() - msecs, hour, min, sec, msec);
        wprintf(L"training time: %02d:%02d:%02d:%03d\n", hour, min, sec, msec);

        if (!ann->save(argv[2]))
                wprintf(L"failed to save %s\n", argv[2]);        





//testing on maxacur and trained network/////////////////////////////////////////////////////////////////////////////////////////////
        ann = new ANNetwork(L"maxacur.nn");       //validate(...)  uses *ann network
        if (!ann->status()) {                     //classification results for maxacur.nn network
                wprintf(L"\nclassification results: maxacur.nn\n");
                validate(&trnrec, TH, &acur, &pacur);
                wprintf(L" \n train set: %d %d\n   sensitivity: %.2f\n   specificity: %.2f\n   +predictive: %.2f\n   -predictive: %.2f\n      accuracy: %.2f\n", trnrec.indices[0].size(), trnrec.indices[1].size(), pacur.se, pacur.sp, pacur.pp, pacur.np, pacur.ac);
                if (vldrec.entries.size()) {
                        validate(&vldrec, TH, &acur, &pacur);
                        wprintf(L" \n validation set: %d %d\n   sensitivity: %.2f\n   specificity: %.2f\n   +predictive: %.2f\n   -predictive: %.2f\n      accuracy: %.2f\n", vldrec.indices[0].size(), vldrec.indices[1].size(), pacur.se, pacur.sp, pacur.pp, pacur.np, pacur.ac);
                }
                if (tstrec.entries.size()) {
                        validate(&tstrec, TH, &acur, &pacur);
                        wprintf(L" \n test set: %d %d\n   sensitivity: %.2f\n   specificity: %.2f\n   +predictive: %.2f\n   -predictive: %.2f\n      accuracy: %.2f\n", tstrec.indices[0].size(), tstrec.indices[1].size(), pacur.se, pacur.sp, pacur.pp, pacur.np, pacur.ac);
                }
        } else
                wprintf(L"failed to load maxacur.nn for classification\n");

        ann = new ANNetwork(argv[2]);         //validate(...)  uses *ann network
        if (!ann->status()) {                 //classification results for trained network
                wprintf(L"\nclassification results: %s\n", argv[2]);
                validate(&trnrec, TH, &acur, &pacur);
                wprintf(L" \n train set: %d %d\n   sensitivity: %.2f\n   specificity: %.2f\n   +predictive: %.2f\n   -predictive: %.2f\n      accuracy: %.2f\n", trnrec.indices[0].size(), trnrec.indices[1].size(), pacur.se, pacur.sp, pacur.pp, pacur.np, pacur.ac);
                if (vldrec.entries.size()) {
                        validate(&vldrec, TH, &acur, &pacur);
                        wprintf(L" \n validation set: %d %d\n   sensitivity: %.2f\n   specificity: %.2f\n   +predictive: %.2f\n   -predictive: %.2f\n      accuracy: %.2f\n", vldrec.indices[0].size(), vldrec.indices[1].size(), pacur.se, pacur.sp, pacur.pp, pacur.np, pacur.ac);
                }
                if (tstrec.entries.size()) {
                        validate(&tstrec, TH, &acur, &pacur);
                        wprintf(L" \n test set: %d %d\n   sensitivity: %.2f\n   specificity: %.2f\n   +predictive: %.2f\n   -predictive: %.2f\n      accuracy: %.2f\n", tstrec.indices[0].size(), tstrec.indices[1].size(), pacur.se, pacur.sp, pacur.pp, pacur.np, pacur.ac);
                }
        } else
                wprintf(L"failed to load %s for classification\n", argv[2]);


}
////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
// for 2 classes only    for N classes in SOM project
void set_validation(PREC vld, PREC trn, float p)
{
        int c1 = int((p / 100.0f) * (float)trn->indices[0].size());
        int c2 = int((p / 100.0f) * (float)trn->indices[1].size());
        wprintf(L" validaton size: %d %d\n", c1, c2);
        if (c1 < 1 || c2 < 1) {
                wprintf(L" validaton is not set, one of the vld class of 0 lenght\n");
                return;
        }

        vld->entries.resize(c1 + c2);

        vld->clsnum.push_back(trn->clsnum[0]);
        vld->clsnum.push_back(trn->clsnum[1]);


        //random shuffle indeces and take first c1,c2 ones//////////////////////
        random_shuffle(trn->indices[0].begin(), trn->indices[0].end());
        random_shuffle(trn->indices[1].begin(), trn->indices[1].end());
        ////////////////////////////////////////////////////////////////////////


        //class1////////////////////////////////////////////
        vector<int> indices;
        indices.resize(c1);
        vld->indices.push_back(indices);
        //get random % from trn set
        for (int i = 0; i < c1; i++) {
                int ind = trn->indices[0].at(i);

                vld->indices[0].at(i) = i;
                vld->entries[i] = trn->entries[ ind ];
                trn->entries[ ind ] = 0;
        }
        trn->indices[0].erase(trn->indices[0].begin(), trn->indices[0].begin() + c1);

        //class2////////////////////////////////////////////
        indices.resize(c2);
        vld->indices.push_back(indices);
        //get random % from trn set
        for (int i = 0; i < c2; i++) {
                int ind = trn->indices[1].at(i);

                vld->indices[1].at(i) = i + c1;
                vld->entries[i+c1] = trn->entries[ ind ];
                trn->entries[ ind ] = 0;
        }
        trn->indices[1].erase(trn->indices[1].begin(), trn->indices[1].begin() + c2);
}

////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
/*
    acur - metric results
    pacur - classification results Se,Sp,...
*/
void validate(PREC rec, float TH, float *acur, PACUR pacur)
{
        float mse = 0.0f, se = 0.0f, sp = 0.0f, pp = 0.0f, np = 0.0f, ac = 0.0f, b;
        float *ivec;
        float ovec[1] = {0.0f};


        int TP = 0, FN = 0, TN = 0, FP = 0;
        int size = (int)rec->entries.size();
        ////////////////////TESTING////////////////////////////////////////////////////////////////
        for (int i = 0; i < (int)rec->entries.size(); i++) {
                //in train set might be 0 entries after set_validation()
                if (rec->entries[i] == 0) {
                        size--;
                        continue;
                }

                ivec = rec->entries[i]->vec;
                ann->classify(ivec, ovec);

                int clstype = (ovec[0] > TH) ? 1 : 2;
                int vcls = rec->entries[i]->cls;

                if (vcls) { //if 1 or 2, 0 if no class info
                        //mse
                        if (vcls == 1)
                                mse += (0.9f - ovec[0]) * (0.9f - ovec[0]);
                        else if (vcls == 2)
                                mse += (0.1f - ovec[0]) * (0.1f - ovec[0]);

                        //se,sp,...
                        if (vcls == clstype) {
                                if (clstype == 1)
                                        TP++;
                                else if (clstype == 2)
                                        TN++;
                        } else { /////error//////////
                                if (clstype == 2 && vcls == 1)  //ill defined as healthy
                                        FN++;
                                else if (clstype == 1 && vcls == 2)  //healthy defined as ill
                                        FP++;
                        }
                }
        }
        mse /= (float)size;   // - 0 marked classes nums
        ///////////////////////////////////////////////////////////////////////////////////////////

        if (TP)
                se = float(TP) / float(TP + FN);
        if (TN)
                sp = float(TN) / float(TN + FP);
        if (TP)
                pp = float(TP) / float(TP + FP);
        if (TN)
                np = float(TN) / float(TN + FN);
        if (TP || FP || FN || TN)
                ac = float(TP + TN) / float(TP + FN + TN + FP);

        pacur->se = 100.0f * se;
        pacur->sp = 100.0f * sp;
        pacur->pp = 100.0f * pp;
        pacur->np = 100.0f * np;
        pacur->ac = 100.0f * ac;

        switch (validation_type) {
        default:
        case 0:
                *acur = 1.0f / mse;    //mse
                break;
        case 1:
                *acur = ac;            //acur
                break;
        case 2:                 //geometric mean se,sp
                *acur = gmean(se * sp, 2);
                break;
        case 3:                 //geometric mean se,pp
                *acur = gmean(se * pp, 2);
                break;
        case 4:                 //geometric mean se,sp,ac
                *acur = gmean(se * sp * ac, 3);
                break;
        case 5:                 //geometric mean se,sp,pp,np,ac
                *acur = gmean(se * sp * pp * np * ac, 5);
                break;
        case 6:                 //F-measure  b=1
                b = 1.0f;
                if (pp && se)
                        *acur = ((b * b + 1) * se * pp) / (b * b * pp + se);
                else
                        *acur = 0;
                break;
        case 7:                 //F-measure  b=1.5
                b = 1.5f;
                if (pp && se)
                        *acur = ((b * b + 1) * se * pp) / (b * b * pp + se);
                else
                        *acur = 0;
                break;
        case 8:                 //F-measure  b=3
                b = 3.0f;
                if (pp && se)
                        *acur = ((b * b + 1) * se * pp) / (b * b * pp + se);
                else
                        *acur = 0;
                break;
        }

}
////////////////////////////////////////////////////////////////////////////////////////////////
float gmean(float m, int n)
{
        return pow(m, 1.0f / (float)n);
}
////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
void dump_sets(PREC trn, PREC vld, PREC tst)
{
        wchar_t name[_MAX_PATH] = L"";
        wchar_t dir[_MAX_PATH] = L"";

        FILE* fp = _wfopen(L"dbgsets.txt", L"wt");

        if (trn != 0) {
                size_t s = 0;
                for (size_t i = 0; i < trn->entries.size(); i++) {
                        if (trn->entries[i] != 0) s++;
                }

                fwprintf(fp, L"TRAINING SET: %d\n", s);
                if (trn->entries.size() < 1000) {
                        for (size_t i = 0; i < trn->entries.size(); i++) {
                                if (trn->entries[i] != 0)  //in train set might be 0 entries after setvld()
                                        fwprintf(fp, L"%s  %d\n", trn->entries[i]->fname, trn->entries[i]->cls);
                        }
                }
        }

        if (vld != 0) {
                fwprintf(fp, L"\n\nVALIDATION SET: %d\n", vld->entries.size());
                if (vld->entries.size() < 1000) {
                        for (size_t i = 0; i < vld->entries.size(); i++)
                                fwprintf(fp, L"%s  %d\n", vld->entries[i]->fname, vld->entries[i]->cls);
                }
        }

        if (tst != 0) {
                fwprintf(fp, L"\n\nTEST SET: %d\n", tst->entries.size());
                if (tst->entries.size() < 1000) {
                        for (size_t i = 0; i < tst->entries.size(); i++)
                                fwprintf(fp, L"%s  %d\n", tst->entries[i]->fname, tst->entries[i]->cls);
                }
        }

        fclose(fp);
}
////////////////////////////////////////////////////////////////////////////////////////////////






////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////TESTING//////////////////////////////////////////////////////////////
void test(int argc, wchar_t* argv[])
{

        REC tstrec;


////parse optional arguments 4,5////////////////////////////////////
        float TH = 0.5f;
        if (argc >= 4 + 1)
                TH = (float)_wtof(argv[4]);
        if (argc >= 5 + 1)
                normalization = _wtoi(argv[5]);
////////////////////////////////////////////////////////////////////


        wprintf(L"loading data...\n");
        FILE *cls1 = _wfopen(argv[3], L"rt");

        if (!cls1) {
                wprintf(L"failed to open file: %s\n", argv[3]);
                exit(1);
        } else
                read_class(cls1, &tstrec);



        if (!tstrec.entries.size()) {
                wprintf(L" no files loaded from: %s.\n", argv[3]);
                exit(1);
        } else
                wprintf(L" %d  files loaded.  size: %d samples\n", tstrec.entries.size(), tstrec.entries[0]->size);



        ann = new ANNetwork(argv[2]);
        if (ann->status()) {
                wprintf(L"failed to load network: %s\n", argv[2]);
                exit(1);
        }
        if (ann->get_layer(0)->get_neurons_number() != vector_length) {
                if (ann->get_layer(0)->get_neurons_number() > vector_length) {
                        wprintf(L" input layer neurons %d are more than data dimension %d", ann->get_layer(0)->get_neurons_number(), vector_length);
                        exit(1);
                } else
                        wprintf(L" input layer neurons %d are less than data dimension %d\n", ann->get_layer(0)->get_neurons_number(), vector_length);
        }
        wprintf(L"%s\n", argv[2]);


        wchar_t name[_MAX_PATH] = L"";
        wchar_t dir[_MAX_PATH] = L"";
        float *ivec;
        float ovec[1] = {0.0f};

        int TP = 0, FN = 0, TN = 0, FP = 0;
        ////////////////////TESTING////////////////////////////////////////////////////////////////
        for (int i = 0; i < (int)tstrec.entries.size(); i++) {

                ivec = tstrec.entries[i]->vec;
                ann->classify(ivec, ovec);


                if (parse_path(tstrec.entries[i]->fname, dir, name))
                        wprintf(L" %s\n", dir);
                int clstype = (ovec[0] > TH) ? 1 : 2 ;
                wprintf(L"  %s   %f   cls %d  ", name, ovec[0], clstype);

                if (tstrec.entries[i]->cls) { //if 1 or 2, 0 if no class info
                        if (tstrec.entries[i]->cls == clstype) {
                                if (clstype == 1)
                                        TP++;
                                else if (clstype == 2)
                                        TN++;
                                wprintf(L"+\n");
                        } else { /////error//////////
                                wprintf(L"-\n");

                                if (clstype == 2 && tstrec.entries[i]->cls == 1)  //ill defined as healthy
                                        FN++;
                                else if (clstype == 1 && tstrec.entries[i]->cls == 2)  //healthy defined as ill
                                        FP++;
                        }
                } else
                        wprintf(L"\n");
        }
        ///////////////////////////////////////////////////////////////////////////////////////////
        if (TP)
                wprintf(L"   sensitivity: %.2f\n", 100.0f * float(TP) / float(TP + FN));
        if (TN)
                wprintf(L"   specificity: %.2f\n", 100.0f * float(TN) / float(TN + FP));
        if (TP)
                wprintf(L"   +predictive: %.2f\n", 100.0f * float(TP) / float(TP + FP));
        if (TN)
                wprintf(L"   -predictive: %.2f\n", 100.0f * float(TN) / float(TN + FN));
        if (TP || FP || FN || TN)
                wprintf(L"      accuracy: %.2f\n", 100.0f * float(TP + TN) / float(TP + FN + TN + FP));

}
////////////////////////////////////////////////////////////////////////////////////////////////





/////////////////////get normalization params////////////////////////////////////////////////////////////
void set_normalization(REC *rec, ANNetwork *pann)
{
        int N = int(rec->entries.size());
        int I = -1;  //first nonzero entry
        for (int i = 0; i < (int)rec->entries.size(); i++) {
                if (rec->entries[i] == 0)
                        N--;
                else if (I == -1)
                        I = i;
        }

        float tmp, min, max;
        vector<float> vmean, vdisp, vmin, vmax;

        //////////get mean, max,min of every feature////////////
        for (int x = 0; x < vector_length; x++) {
                float *ivec = rec->entries[I]->vec;
                tmp = 0.0f;
                min = ivec[x];
                max = ivec[x];
                for (int y = 0; y < (int)rec->entries.size(); y++) {
                        if (rec->entries[y] == 0) continue;

                        ivec = rec->entries[y]->vec;
                        tmp += ivec[x];

                        if (ivec[x] > max) max = ivec[x];
                        if (ivec[x] < min) min = ivec[x];
                }

                vmean.push_back(tmp / float(N));
                vmax.push_back(max);
                vmin.push_back(min);
        }

        ///////get std of every feature////////////////////////
        for (int x = 0; x < vector_length; x++) {
                float *ivec;
                tmp = 0.0f;
                for (int y = 0; y < (int)rec->entries.size(); y++) {
                        if (rec->entries[y] == 0) continue;

                        ivec = rec->entries[y]->vec;
                        tmp += (ivec[x] - vmean[x]) * (ivec[x] - vmean[x]);
                }

                tmp = sqrt(tmp / float(N - 1));
                vdisp.push_back(tmp);
        }



        ///////write normalization coeffs to input layer///////
        for (int n = 0; n < pann->get_layer(0)->get_neurons_number(); n++) {
                pann->get_layer(0)->get_neuron(n)->get_input_link(0)->set_add_term(-vmin[n]);
                switch (normalization) {
                case MINMAX:
                        pann->get_layer(0)->get_neuron(n)->set_function(ANeuron::LINEAR);                        
                        if (fabs(float(vmax[n] - vmin[n])) != 0.0f)
                                pann->get_layer(0)->get_neuron(n)->get_input_link(0)->set_weight(1.0f / (vmax[n] - vmin[n]));                                
                        break;

                default:
                case ZSCORE:  //zscore
                        pann->get_layer(0)->get_neuron(n)->set_function(ANeuron::LINEAR);                                                
                        if (vdisp[n] != 0.0f)
                                pann->get_layer(0)->get_neuron(n)->get_input_link(0)->set_weight(1.0f / vdisp[n]);                                
                        break;

                case SIGMOIDAL:  //sigmoidal
                        pann->get_layer(0)->get_neuron(n)->set_function(ANeuron::SIGMOID);                                                
                        if (vdisp[n] != 0.0f)
                                pann->get_layer(0)->get_neuron(n)->get_input_link(0)->set_weight(1.0f / vdisp[n]);                                
                        break;
                }
        }
        ///////write normalization coeffs to input layer///////

}
///////////////////////////////////////////////////////////////////////////////////








////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////data loading routines/////////////////////////////////////////////////////////
int read_line(FILE *f, wchar_t *buff, int *c)
{
        wint_t res = 0;
        wchar_t *pbuff = buff;

        while ((short)res != EOF) {
                res = fgetwc(f);
                if (res == 0xD || res == 0xA) {
                        if (pbuff == buff) continue;

                        *pbuff = 0;
                        if (!c) {
                                return 1;
                        } else {
                                int ptr = (int)wcslen(buff) - 1;

                                while (ptr > 0) {  //skip thru 'spaces'      dir/asds.f ___1__ \n
                                        if (buff[ptr] != 0x20) break;
                                        else ptr--;
                                }
                                while (ptr > 0) {  //skip thru 'clas type'
                                        if (buff[ptr] == 0x20) break;
                                        else ptr--;
                                }

                                if (ptr) {
                                        *c = _wtoi(&buff[ptr+1]);
                                        while (buff[ptr] == 0x20)  //remove blanks from end of string
                                                buff[ptr--] = 0;
                                } else
                                        *c = 0;

                                return 1;
                        }
                }
                if ((short)res != EOF) {
                        *pbuff++ = (char)res;
                }
        }

        return (short)res;
}

/*
    format 1           //data stored in separate files: ECG,FOUR
     file1 [class]
     file2 [class]
     ...

    format 2           //data stored in this file
     file1 [class]
      vec1 ...
     file2 [class]
      vec1 ...
     ...

            */
/*
     read class data to PTSTREC struct
                                        */
void read_class(FILE *fp, PREC rec, int c)
{
        wchar_t ustr[_MAX_PATH], *pstr;
        int res = 1, cls;

        int entrsize = (int)rec->entries.size();   //size from previous read iteration

        while (res > 0) {
                res = read_line(fp, ustr, &cls);
                if (res > 0) {
                        if (c && !cls) //put default if (c=1,2 and cls=0)
                                cls = c;


                        CSignal *sig = new CSignal(ustr);

                        if (sig->N && sig->M) {   //read file FORMAT 1.*
                                if (!vector_length)
                                        vector_length = sig->M;
                                else {
                                        if (vector_length != sig->M) {
                                                wprintf(L"fmt1.*: vector %s (lenght %d) is not equal to vlen: %d", ustr, sig->M, vector_length);
                                                exit(1);
                                        }
                                }

                                for (int j = 0; j < sig->N; j++) {
                                        if (normalization == 4)
                                                sig->nenergy(sig->data[j], vector_length);
                                        if (normalization == 5)
                                                sig->nminmax(sig->data[j], vector_length, 0.1f, 0.9f);

                                        PENTRY entry = new ENTRY;
                                        entry->vec = sig->data[j];
                                        entry->size = vector_length;
                                        swprintf(entry->fname, L"%s_%d", ustr, j);
                                        entry->cls = cls;
                                        rec->entries.push_back(entry);
                                }

                                signals.push_back(sig);
                        }

                        else {  //FORMAT 2
                                //[filename] [class]
                                //samples
                                float tmp;
                                vector<float> fvec;

                                while (fwscanf(fp, L"%f", &tmp) == 1)
                                        fvec.push_back(tmp);

                                if (fvec.size() == 0) {
                                        wprintf(L"fmt2: vector %s has zero lenght", ustr);
                                        exit(1);
                                }

                                if (!vector_length)
                                        vector_length = (int)fvec.size();
                                else {
                                        if (vector_length != (int)fvec.size()) {
                                                wprintf(L"fmt2: vector %s (lenght %d) is not equal to vector_length: %d", ustr, fvec.size(), vector_length);
                                                exit(1);
                                        }
                                }

                                pstr = new wchar_t[_MAX_PATH];
                                wcscpy(pstr, ustr);

                                if (normalization == 4)
                                        sig->nenergy(&fvec[0], vector_length);
                                if (normalization == 5)
                                        sig->nminmax(&fvec[0], vector_length, 0.1f, 0.9f);

                                float *fdata = new float[vector_length];
                                for (int i = 0; i < vector_length; i++)
                                        fdata[i] = fvec[i];

                                PENTRY entry = new ENTRY;
                                entry->vec = fdata;
                                entry->size = vector_length;
                                wcscpy(entry->fname, pstr);
                                entry->cls = cls;
                                rec->entries.push_back(entry);


                                delete sig;
                        }

                }// if(res > 0) line was read from file
        }// while(res > 0)  res = read_line(fp,ustr, &cls);
        fclose(fp);


        //arrange indices of classes
        if ((int)rec->entries.size() > entrsize) {
                //find new classes in entries not in rec->clsnum array
                for (int i = entrsize; i < (int)rec->entries.size(); i++) {
                        int cls = rec->entries[i]->cls;
                        bool match = false;
                        for (int j = 0; j < (int)rec->clsnum.size(); j++) {
                                if (cls == rec->clsnum[j]) {
                                        match = true;
                                        break;
                                }
                        }
                        if (!match) //no match
                                rec->clsnum.push_back(cls);
                }
                //clsnum = [cls 1][cls 2] ... [cls N]   N entries
                //clsnum = [1][2][3] or [3][1][2] or ... may be not sorted


                if (rec->clsnum.size() > rec->indices.size()) {
                        vector<int> indices;
                        int s = (int)(rec->clsnum.size() - rec->indices.size());
                        for (int i = 0; i < s; i++)
                                rec->indices.push_back(indices);
                }
                //arrange indices
                for (int i = 0; i < (int)rec->clsnum.size(); i++) {
                        //fill positions of clsnum[i] class to indices vector
                        for (int j = entrsize; j < (int)rec->entries.size(); j++) {
                                if (rec->clsnum[i] == rec->entries[j]->cls)
                                        rec->indices[i].push_back(j);
                        }
                }
        }
}
//////////////////data loading routines/////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////










////////////////////////////////////////////////////////////////////////////////////////////////
void get_file_name(wchar_t *path, wchar_t *name)
{
        int sl = 0, dot = (int)wcslen(path);
        int i;
        for (i = 0; i < (int)wcslen(path); i++) {
                if (path[i] == '.') break;
                if (path[i] == '\\') break;
        }
        if (i >= (int)wcslen(path)) {
                wcscpy(name, path);
                return;
        }

        for (i = (int)wcslen(path) - 1; i >= 0; i--) {
                if (path[i] == '.')
                        dot = i;
                if (path[i] == '\\') {
                        sl = i + 1;
                        break;
                }
        }

        memcpy(name, &path[sl], (dot - sl)*2);
        name[dot-sl] = 0;
}

int parse_path(wchar_t *path, wchar_t *dir, wchar_t *name)   //true if dirs equal
{
        int res;
        int i;
        for (i = (int)wcslen(path) - 1; i > 0; i--) {
                if (path[i] == '\\')
                        break;
        }

        if (i) { //path + name
                wcscpy(name, &path[i+1]);
                path[i] = 0;
                res = wcscmp(dir, path);
                wcscpy(dir, path);
        } else { //no path
                res = wcscmp(dir, L"");
                wcscpy(dir, L"");
                wcscpy(name, path);
        }
        return res;   //res=0 if dir and path\filename are equal
}
//////////////////////////////////////////////////////////////////////////////////////////


void msec_to_time(int msec, int& h, int& m, int& s, int& ms)
{
        ms = msec % 1000;
        msec /= 1000;

        if (msec < 60) {
                h = 0;
                m = 0;
                s = msec;                 //sec to final
        } else {
                float tmp;
                tmp = (float)(msec % 60) / 60;
                tmp *= 60;
                s = int(tmp);
                msec /= 60;

                if (msec < 60) {
                        h = 0;
                        m = msec;
                } else {
                        h = msec / 60;
                        tmp = (float)(msec % 60) / 60;
                        tmp *= 60;
                        m = int(tmp);
                }
        }
}
