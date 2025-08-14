user = 'sebas'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split


from rdkit.Chem import Descriptors

data_path = "C:\\Users\\sebas\\OneDrive - University of Leeds\\PhD\\Paper-Chapter 2\\Workflow\\Data\\"
fig_path = "C:\\Users\\sebas\\OneDrive - University of Leeds\\PhD\\Paper-Chapter 2\\Workflow\\Figs\\"



def Matrix_averaging(X_df, m0 = None):

    '''
    Function will either number average the matrix X or perform a weighted average by molar mass.
    Args:
    X_df - dataframe contains all polymer names by index and all edge params in the columns
    m0 - the molar mass vector of all edge params
    
    Returns:
    number or molar mass averged matrix X
    '''
    
    
    
    X = np.array(X_df)

    if m0 is not None:
        m0.reindex(X_df.columns)
        Mp = np.diag(m0)
        Mn = np.diag((X @ m0.values).reshape(-1,))
        X_mav = np.linalg.inv(Mn) @ X @ Mp
        Xn = pd.DataFrame(X_mav, columns=X_df.columns, index=X_df.index)
        
        
    else:
        X = X / np.linalg.norm(X, ord=1, axis=1, keepdims=True)
        Xn = pd.DataFrame(X, columns=X_df.columns, index=X_df.index) # df with named col and numbered index

    return Xn




class GetData:
    
    def __init__(self, keep_poly_ID=False, drop_obs=False):
        self.keep_poly_ID = keep_poly_ID
        
        self.names = None
        
        # self.cols_to_drop = None
    
        poly_frag_raw = pd.read_excel(data_path+'data.xlsx', engine='openpyxl', skiprows=2)
        frag_rigid_raw = pd.read_excel(data_path+'frag_rigid_matrix.xlsx', engine='openpyxl', skiprows=2)
        frag_sidechain_raw = pd.read_excel(data_path+'frag_sidechain_matrix.xlsx', engine='openpyxl', skiprows=2)
        
        smiles_frag_ID_raw = pd.read_excel(data_path+'data.xlsx', engine='openpyxl', nrows=2)
        smiles_rigid_ID_raw = pd.read_excel(data_path+'frag_rigid_matrix.xlsx', engine='openpyxl', nrows=2)
        smiles_sidechain_ID_raw = pd.read_excel(data_path+'frag_sidechain_matrix.xlsx', engine='openpyxl', nrows=2)

        self.smiles_frag_ID = pd.Series(smiles_frag_ID_raw.values[0,:].tolist(), index=smiles_frag_ID_raw.values[1,:].tolist())['S':]
        self.smiles_rigid_ID = pd.Series(smiles_rigid_ID_raw.values[0,:].tolist(), index=smiles_rigid_ID_raw.values[1,:].tolist())['S':]
        self.smiles_sidechain_ID = pd.Series(smiles_sidechain_ID_raw.values[0,:].tolist(), index=smiles_sidechain_ID_raw.values[1,:].tolist())['_4':]

        self.poly_frag_all = poly_frag_raw.dropna(subset=['Tg1']).reset_index(drop=True)
        if drop_obs:
            subset = ['T_2_(10-10)2_(10-10) S S', 'T_3_(10-10)3_(10-10) S S', 'T_5_(10-10)5_(10-10) S S', 'M_2_(6-8)2_(6-8) S', 'M_2_(8-10)2_(8-10) S', 'M_2_(10-12)2_(10-12) S', 'M_2_(12-14)2_(12-14) S']
            self.poly_frag_all = self.poly_frag_all[~self.poly_frag_all.poly_ID.isin(subset)]
        
        self.poly_frag, self.poly_frag_val = train_test_split(self.poly_frag_all, test_size=0.25, random_state=42)
        self.frag_rigid = frag_rigid_raw
        self.frag_sidechain = frag_sidechain_raw

        self.frag_cols = self.poly_frag.loc[:,'S':].columns.sort_values()

    def _end_cap(self, smiles):
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        smiles= smiles.apply(lambda x: x.replace('([*])', ''))
        smiles = smiles.apply(lambda x: x.replace('[*]/', ''))
        smiles = smiles.apply(lambda x: x.replace('/[*]', ''))
        smiles = smiles.apply(lambda x: x.replace('\[*]', ''))
        smiles = smiles.apply(lambda x: x.replace('[*]', ''))
        return smiles

    def get_poly_smiles(self, end_cap):
        poly_smiles = self.poly_frag.SMILES
        if end_cap==True:
            poly_smiles = self._end_cap(poly_smiles)
        return poly_smiles

    def get_poly_smiles_val(self, end_cap):
        poly_smiles = self.poly_frag_val.SMILES
        if end_cap==True:
            poly_smiles = self._end_cap(poly_smiles)
        return poly_smiles
    
    def get_poly_smiles_all(self, end_cap, trade_names=False):
        poly_smiles = self.poly_frag_all.SMILES
        if end_cap==True:
            poly_smiles = self._end_cap(poly_smiles)
        if trade_names:
            poly_smiles=pd.Series(poly_smiles)
            poly_smiles.index=self.poly_frag_all.poly_name
        return poly_smiles
    
    def get_frag_smiles(self, end_cap):
        frag_smiles =  self.smiles_frag_ID
        if end_cap==True:
            frag_smiles = self._end_cap(frag_smiles)
        return frag_smiles
    
        
    def get_rigid_smiles(self, end_cap=False):
        rigid_smiles = self.smiles_rigid_ID
        if end_cap==True:
            rigid_smiles = self._end_cap(rigid_smiles)
        return rigid_smiles

    def get_sidechain_smiles(self, end_cap=False):
        sidechain_smiles = self.smiles_sidechain_ID
        if end_cap==True:
            sidechain_smiles = self._end_cap(sidechain_smiles)
        return sidechain_smiles

    ##### Composition matrices #####
    
    def get_X(self):
        X = self.poly_frag.loc[:,'S':self.poly_frag.columns[-1]] # range of full fragmnets/ basis
        X.fillna(0, inplace=True)
        if self.keep_poly_ID:
            X.index = self.poly_frag.poly_ID
        return X

    def get_X_val(self):
        X = self.poly_frag_val.loc[:,'S':self.poly_frag.columns[-1]] # range of full fragmnets/ basis
        X.fillna(0, inplace=True)
        if self.keep_poly_ID:
            X.index = self.poly_frag_val.poly_ID
        return X

    def get_X_all(self, poly_names=False):
        X = self.poly_frag_all.loc[:,'S':self.poly_frag.columns[-1]] # range of full fragmnets/ basis
        X.fillna(0, inplace=True)
        if self.keep_poly_ID:
            X.index = self.poly_frag_all.poly_ID

        return X

    def get_A(self):
        A = self.frag_rigid.loc[:,'S':] # range of full fragmnets/ basis
        A.fillna(0, inplace=True)
        A.index = self.frag_rigid.Name
        return A


    def get_W(self):
        W = self.frag_sidechain.loc[:,'_4':] # range of full fragmnets/ basis
        W.fillna(0, inplace=True)
        W.index = self.frag_sidechain.Name
        return W

    
    ##### Descriptor matrices #####
    
    def get_D_poly(self):
        with open(data_path+'Descriptor_poly_matrix.pkl', 'rb') as f:
            D_raw = pickle.load(f)
        D = pd.DataFrame(data=D_raw.loc[:, 'ABC':])
        D.index = D_raw.Name
        return D
    
    def get_D_poly_val(self):
        with open(data_path+'Descriptor_poly_matrix_val.pkl', 'rb') as f:
            D_raw = pickle.load(f)
        D = pd.DataFrame(data=D_raw.loc[:, 'ABC':])
        D.index = D_raw.Name
        return D
    
        
    def get_D_frag(self):
        with open(data_path+'Descriptor_frag_matrix.pkl', 'rb') as f:
            D_raw = pickle.load(f)
        D = pd.DataFrame(data=D_raw.loc[:, 'ABC':])
        D.index = D_raw.Name
        return D
    
    
    def get_D_rigid(self):
        with open(data_path+'Descriptor_rigid_matrix.pkl', 'rb') as f:
            D_raw = pickle.load(f)
        D = pd.DataFrame(data=D_raw.loc[:, 'ABC':])
        D.index = D_raw.Name
        return D

    def get_D_sidechain(self):
        with open(data_path+'Descriptor_sidechain_matrix.pkl', 'rb') as f:
            D_raw = pickle.load(f)
        D = pd.DataFrame(data=D_raw.loc[:, 'ABC':])
        D.index = D_raw.Name
        return D

    
    def get_Tg(self):
        y_range = self.poly_frag.loc[:, 'Tg1':'Tg6']
        y_avg = y_range.mean(axis=1, skipna=True) + 273.15
        if self.keep_poly_ID == True:
            y_avg = pd.Series(y_avg)
            y_avg.index=self.poly_frag.poly_ID
        return y_avg

    def get_Tg_val(self):
        y_range = self.poly_frag_val.loc[:, 'Tg1':'Tg6']
        y_avg = y_range.mean(axis=1, skipna=True) + 273.15
        if self.keep_poly_ID == True:
            y_avg = pd.Series(y_avg)
            y_avg.index=self.poly_frag_val.poly_ID
        return y_avg

    def get_Tg_all(self):
        y_range = self.poly_frag_all.loc[:, 'Tg1':'Tg6']
        y_avg = y_range.mean(axis=1, skipna=True) + 273.15
        if self.keep_poly_ID == True:
            y_avg = pd.Series(y_avg)
            y_avg.index=self.poly_frag.poly_ID
        return y_avg

    def get_poly_names(self):
        return self.poly_frag_all.loc[:,'poly_name']

    # ------------------------
    # think this is old code:
    # ------------------------
    
    # def get_D(self, just_3D=True):
        
    #     if just_3D:
    #         with open(data_path+'mordred_3D.pkl', 'rb') as f:
    #             self.df = pickle.load(f)
    #         D_df = self.df.loc[:,'PNSA1':] # 3D
    #         D_df.drop(columns=['tail_mw'], inplace=True) # 3D
            
    #     else:
    #         with open(data_path+'mordred_all.pkl', 'rb') as f:
    #             self.df = pickle.load(f)
    #         D_df = self.df.loc[:,'ABC':]
    #         D_df.drop(columns=['tail_mw', 'MW'], inplace=True)
        
        
    #     self.df['rigid_mw'] = self.df.Mol.apply(lambda x : Descriptors.ExactMolWt(x))
         
    #     D_df['w_frac'] = self.df.tail_mw/(self.df.rigid_mw + self.df.tail_mw)
        
    #     if self.o_set==False:
    #         D_df = D_df.loc[~self.cols_to_drop, :]
    #         self.df = self.df.loc[~self.cols_to_drop, :]
        
    #     return D_df

    # ------------------------
    
    # other studies
    def filter_Xie_mask(self):
        refs_included = [1, 4, 9, 43, 34, 17, 15]
        ref_mask = self.poly_frag_all.loc[:,'ref1':'ref6'].isin(refs_included)
        ref_mask.columns = self.poly_frag.loc[:,'Tg1':'Tg6'].columns
        return ref_mask

    def get_Xie_Tg(self):
        mask = self.filter_Xie_mask()
        y_range = self.poly_frag_all.loc[:, 'Tg1':'Tg6']
        y_avg = y_range[mask].mean(axis=1, skipna=True).dropna() + 273.15
        # y_avg = y_range[mask]
        if self.keep_poly_ID == True:
            y_avg = pd.Series(y_avg)
            y_avg.index=self.poly_frag.poly_ID
        return y_avg
    

    

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))

