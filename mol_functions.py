# basic modules
import numpy as np
import pandas as pd
from tqdm import tqdm


from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdDepictor, rdAbbreviations, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina


from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.drawOptions.addAtomIndices = True
# IPythonConsole.drawOptions.addStereoAnnotation = False
IPythonConsole.molSize = 300,230
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO







def get_highlight_atom_colors(atoms, highlightAtomColors={}, color="red"):
    colors = {
        "blue": (0.6, 0.8, 1.0),
        "green": (0.7, 1.0, 0.7),
        "pink": (1.0, 0.8, 0.9),
        "yellow": (1.0, 1.0, 0.6),
        "orange": (1.0, 0.8, 0.6),
        "lavender": (0.8, 0.7, 1.0),
        "mint": (0.7, 1.0, 0.9),
        "red": (1.0, 0.6, 0.6)
        }
    
    highlightAtomColors.update({
        atom: colors[color] for atom in atoms
    })
    return highlightAtomColors








class DrawMols:
    
    def __init__(self, ppi=500, fontsize=10, bond_length=1.5, abbrevs=True, reset_coords=True):
        """
        Class for consistently drawing polymers using RDKit using ACS 1996 style.
        """
        self.ppi = ppi
        self.fontsize=fontsize
        self.bond_length=bond_length
        self.abbrevs = abbrevs
        self.reset_coords = reset_coords
        self.sf = self.ppi/96
        
        if abbrevs:
            self.nabbrevs = self._generate_abbreviations(default=False)


    def fill_abbrevs(self, mol, maxCoverage=1):
        rdDepictor.Compute2DCoords(mol)
        rdDepictor.StraightenDepiction(mol)
        mol = rdAbbreviations.CondenseMolAbbreviations(mol, self.nabbrevs, maxCoverage=maxCoverage)
        rdDepictor.Compute2DCoords(mol)
        rdDepictor.StraightenDepiction(mol)
        return mol
        

    def get_dopts(self, dopts=None):
        if dopts is None:
            d2d = Draw.MolDraw2DCairo(-1,-1)
            dopts = d2d.drawOptions()
               
        
        b = self.bond_length/self.sf
        w = 0.6*self.sf
        s = round(10*self.sf)
        
        Draw.SetACS1996Mode(dopts, b)
        dopts.dummiesAreAttachments = True
        dopts.atomHighlightsAreCircles = True
        # Customize options
        dopts.bondLineWidth = 0.6*self.sf
        # dopts.scaleBondWidth = True
        dopts.scalingFactor = 14.4 /  b
        dopts.multipleBondOffset = 0.18
        dopts.highlightBondWidthMultiplier = round(32/self.sf)
        dopts.fixedFontSize = round(self.fontsize*self.sf)
        dopts.dummiesAreAttachments = True
        dopts.legendFontSize = round(self.fontsize * self.sf * 1.2) 
        
        return dopts
        


    def show_mol(self, mol, maxCoverage=1, dopts=None, atomNumber=False, **kwargs):
        """
        Inputs
        ------
        mol : RDKit mol object
        kwargs : RDKit's Chem.Draw arguments
            including args such as highlightAtoms (list), legend (str)
        legend : String : Defaults to empty string
        highlightAtoms : List : Defaults to empty list


        Example useage
        --------------
        dm = DrawMols()
        mol = Chem.MolFromSmiles('[*]C1=C(CCCCCCCCCCCC)C=C([*])S1')
        dm.show_mol(mol, highlightAtoms=[0,1,2,4,5,6,7], legend='PAT')
        """


        # rdDepictor.Compute2DCoords(mol)
        # rdDepictor.StraightenDepiction(mol)
        
        # if abbreviations:
        #     nabbrevs = self._generate_abbreviations(default=False)
        #     mol = rdAbbreviations.CondenseMolAbbreviations(mol, nabbrevs, maxCoverage=maxCoverage)

        if self.abbrevs:
            mol = self.fill_abbrevs(mol, maxCoverage=maxCoverage)
        
        if self.reset_coords:
            rdDepictor.Compute2DCoords(mol)  
            rdDepictor.StraightenDepiction(mol)
        
        d2d = Draw.MolDraw2DCairo(-1,-1)
        dopts = d2d.drawOptions()
        # dopts 
        dopts = self.get_dopts(dopts)

        d2d.DrawMolecule(mol, **kwargs)
        d2d.FinishDrawing()
        bio = BytesIO(d2d.GetDrawingText())
        return Image.open(bio)


    
    def show_mol_row(self, mols, buffer=5, arrange_by_mass=False, legend=None):
        """
        Inputs
        ------
        mols : List
            Each item is an RDKit Mol.
        row_buffer : int, optional
            The vertical space between rows, default is 10.
        col_buffer : int, optional
            The horizontal space between images in a row.
        arrange_by_mass : Bool, optional
            Rearranges the images by their molar mass
        legend : List of strings, optional
            The string attached to each image in the row
            
        
        Returns
        -------
        PIL.Image.Image
            A single image containing the row of images.
        
        Example usage
        -------------
        smileses =['[*]C1=CC=C([*])S1', 
                   '[*]C1=C(CCCC)C=C([*])S1', 
                   '[*]C1=C(CCCCCC)C=C([*])S1', 
                   '[*]C1=C(CCCCCCCC)C=C([*])S1', 
                   '[*]C1=C(CCCCC)C=C([*])S1']
            
        dm = DrawMols()
        mols = [Chem.MolFromSmiles(smiles) for smiles in smileses]
        dm.show_mol_row(mols, arrange_by_mass=True, legend=smileses)
        """

        if legend is None:
            legend=['']*len(mols)
            
        if arrange_by_mass:
            mols_series = self._arrange_mols_by_mass(mols)
            mols = mols_series.tolist()
            legend = [legend[i] for i in mols_series.index]
        
            
        imgs = [self.show_mol(mol, legend=legend_i) for mol, legend_i in zip(mols, legend)]

        return self.show_image_row(imgs, buffer=buffer)

    
    def show_image_row(self, imgs, buffer=5):
        
        height = 0
        width = 0
    
        # Determine the maximum height and total width of the resulting image
        for img in imgs:
            height = max(height, img.height)
            width += img.width
        width += buffer * (len(imgs) - 1)
    
        # Create a new blank image with calculated dimensions
        row_img = Image.new("RGBA", (width, height))
    
        x = 0  # Starting x-coordinate for pasting images
        for img in imgs:
            # Calculate the y-offset to center the image vertically
            y_offset = (height - img.height) // 2
            row_img.paste(img, (x, y_offset))
            x += img.width + buffer  # Update x-coordinate for the next image
    
        return row_img


    
    def show_mol_grid(self, nested_mols, row_buffer=10, col_buffer=5, 
                     arrange_by_mass=False, legend=None, row_labels=None, 
                     row_label_fontsize=None, row_alignment='center'):

        """
        Displays a grid of images from a nested list of image objects with optional row labels and alignment control.
    
        Parameters
        ----------
        nested_mols : list of lists
            Each item is an RDKit Mol.
        row_buffer : int, optional
            The vertical space between rows, default is 10.
        col_buffer : int, optional
            The horizontal space between images in a row.
        arrange_by_mass : Bool, optional
            Rearranges the images by their molar mass
        legend : Nested List containing List of strings, optional
            The string attached to each image in the grid
        row_labels : List, optional (default: None)
            Index label for each row of images, list of strings
        row_alignment : str, optional (default: None)
            'left' or 'center' to force alignment. If None:
            - Left-aligns when row_labels are present
    
        Returns
        -------
        PIL.Image.Image
            A single image containing the grid of images.
    
        Example usage
        -------------
        dm = DrawMols(ppi=300, fontsize=10)
        smiles_nested = [
            ['[*]C1=CC=C([*])S1', '[*]C1=C(CCCC)C=C([*])S1'],
            ['[*]C1=C(CCCCC)C=C([*])S1', '[*]C1=C(CCCCCC)C=C([*])S1', '[*]C1=C(CCCCCCCC)C=C([*])S1']
        ]
        mols_nested = [[Chem.MolFromSmiles(smiles) for smiles in row] for row in smiles_nested]
        dm.show_mol_grid(mols_nested, row_buffer=20, col_buffer=10, row_labels=['I', 'II'], legend=smiles_nested, row_alignment='center')
        
        Parameters
        ----------

        """

        
        if legend is None:
            legend = [[''] * len(mols) for mols in nested_mols]
        
        # Generate row images with labels if specified
        row_images = [
            self.show_mol_row(mols, buffer=col_buffer, 
                             arrange_by_mass=arrange_by_mass, 
                             legend=legend_i)
            for mols, legend_i in zip(nested_mols, legend)
        ]
    
    
        if row_labels is not None:
            if len(row_labels) != len(row_images):
                raise ValueError("row_labels must have the same length as the number of rows in nested_mols")
            
            font_size_pts = round(self.fontsize * self.sf * 1.5) 
            try:
                font = ImageFont.truetype("arial.ttf", font_size_pts)
            except IOError:
                font = ImageFont.load_default()
            
            new_row_images = []
            for label, row_img in zip(row_labels, row_images):
                label_str = str(label)
                text_bbox = font.getbbox(label_str)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                padding = 10 # Horizontal padding around the text
                label_width = text_width + 2 * padding
                label_height = row_img.height
                
                # Create label image with transparent background
                label_img = Image.new("RGBA", (label_width, label_height), (255, 255, 255, 255))
                draw = ImageDraw.Draw(label_img)
                # Draw label text vertically centered
                draw.text((padding, label_height // 2), label_str, font=font, fill="black", anchor="lm")
                
                # Combine label and row image
                combined_row = Image.new("RGBA", (label_img.width + col_buffer + row_img.width, row_img.height))
                combined_row.paste(label_img, (0, 0))
                combined_row.paste(row_img, (label_img.width + col_buffer, 0))
                new_row_images.append(combined_row)
            
            row_images = new_row_images
            
        # Determine alignment logic
        if row_alignment:
            row_alignment = row_alignment.lower()
            if row_alignment not in ('left', 'center'):
                raise ValueError("row_alignment must be 'left' or 'center'")
            left_align = (row_alignment == 'left')
        else:
            left_align = (row_labels is not None)  # Default behavior
    
        # Build final grid
        max_width = max(img.width for img in row_images) if row_images else 0
        total_height = sum(img.height for img in row_images) + row_buffer * (len(row_images)-1)
        grid_img = Image.new("RGBA", (max_width, total_height))
    
        y = 0
        for row_img in row_images:
            x = 0 if left_align else (max_width - row_img.width) // 2
            grid_img.paste(row_img, (x, y))
            y += row_img.height + row_buffer
    
        return grid_img


    
    # Utility functions
    def _arrange_mols_by_mass(self, mols):
        """
        Arranges list of mols by their mass
    
        Parameters
        ----------
        mols : List
            Containing RDKit molecule objects.
    
        Returns
        -------
        List 
            Containing RDKit molecule objects arranged by their molar mass.
        """
    
        df = pd.DataFrame({'mol':mols})
        df['mw'] = df['mol'].apply(lambda x: Chem.rdMolDescriptors.CalcExactMolWt(x))
        sorted_df = df.sort_values('mw')
        return sorted_df['mol']


    def _generate_abbreviations(self, default=True):
        """
        Generates the abbreviations for alkyl tails with rdAbbreviations

        Returns
        -------
        RDKit rdAbbreviations.ParseAbbreviations

        Example useage
        --------------
        dm = DrawMols()
        nabbrevs = dm._generate_abbreviations()
        for nabbrev in nabbrevs:
            print(nabbrev.label,Chem.MolToSmiles(nabbrev.mol))
        smiles = 'COC(C=C(/C=C/[*])C(OCC(CC)CCCC)=C1)=C1/C=C/C2=C(OC)C=C([*])C(OC)=C2'
        mol = Chem.MolFromSmiles(smiles)
        nmol = rdAbbreviations.CondenseMolAbbreviations(mol,nabbrevs)
        nmol
        """

        if default:
            nabbrevs = rdAbbreviations.GetDefaultAbbreviations()
        
        else:
            abbreviations = []
            for n in range(20, 1, -1):
                smiles = "*" + "C" * n  # Generate SMILES for alkyl chain
                abbrevR = f"C<sub>{n}</sub>H<sub>{2 * n + 1}</sub>" 
                abbrevL = f"H<sub>{2 * n + 1}</sub>C<sub>{n}</sub>"
                # abbrevL = f"{2 * n + 1}H{n}C"
                abbreviations.append(f"{abbrevR}    {smiles}    {abbrevR}    {abbrevL}")
    
            abbreviation_string = "\n    ".join(abbreviations)
            nabbrevs = rdAbbreviations.ParseAbbreviations(abbreviation_string)     
        return nabbrevs



    def nest_the_list(self, flat_list, n_cols=5):
        # Calculate the number of rows
        n_rows = (len(flat_list) + n_cols - 1) // n_cols  # Ceiling division
        
        # Create the grid and labels
        grid_list = []
        labels = []
        for row in range(n_rows):
            start_idx = row * n_cols
            end_idx = start_idx + n_cols
            row_list = flat_list[start_idx:end_idx]
            grid_list.append(row_list)
        return grid_list 





class ToSubstructures:

    def __init__(self, mol):
        """
        Takes a molecule and breaks it into the desired substructures based on the atoms/bonds in the main chain and atoms/bonds in the side chains
        
        Args
        ----
        - mol: RDKit Mol object
        
        """

        ########################################################
        # This code will make sure that after breaking up the mol into fragments, the main chain fragment will always be the first in the List/Tuple
        dummy_atom_idx = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':  # Identify the dummy atom
                dummy_atom_idx = atom.GetIdx()
                break

        if dummy_atom_idx is not None:
            # Generate a new atom ordering with the dummy atom first
            new_order = [dummy_atom_idx] + [i for i in range(mol.GetNumAtoms()) if i != dummy_atom_idx]
            # Reorder the molecule's atoms
            mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
        else:
            raise ValueError("No dummy atom ([*]) found in the molecule. This is a functionality requirement.")
        ##########################################################
        
        self.mol = mol
        self.mc_atoms = []
        self.nAts = self.mol.GetNumAtoms()
        self.em = Chem.EditableMol(self.mol)
        self.fragments = ()
        

    def get_self_avoiding_paths(self, atom_i, atom_j):
        """
        Get all self-avoiding paths from atom_i to atom_j in a molecule.
    
        Args
        ----
        - self.mol: RDKit Mol object
        - atom_i: Index of the starting atom
        - atom_j: Index of the target atom
    
        Returns
        -------
        - List of paths, where each path is a list of atom indices.
    
        # Example usage
        ---------------
        mol = Chem.MolFromSmiles('*c1ccc(*)c(*)c1')
        ts = ToSubstructures(mol)
        atom_i = 3  # Starting atom index
        atom_j = 8  # Target atom index
        paths = ts.get_self_avoiding_paths(atom_i, atom_j)
        print("Self-avoiding paths:", paths)
        """
        def dfs(current, target, visited, path, all_paths):
            if current == target:
                all_paths.append(path.copy())
                return
            visited.add(current)
            for neighbor in self.mol.GetAtomWithIdx(current).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    path.append(neighbor_idx)
                    dfs(neighbor_idx, target, visited, path, all_paths)
                    path.pop()  # Backtrack
            visited.remove(current)
    
        visited = set()
        all_paths = []
        dfs(atom_i, atom_j, visited, [atom_i], all_paths)
        return all_paths
    
    def find_dummy_atoms(self):
        """
        Find the atom indices of dummy atoms in a molecule.
    
        Args
        ----
        - self
    
        Returns
        -------
        - List of atom indices that are dummy atoms.
        
        Example usage
        -------------
        mol = Chem.MolFromSmiles('*c1ccc(*)c(*)c1')
        ts = ToSubstructures(mol)
        print(ts.find_dummy_atoms())
        """
        dummy_indices = []
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atoms have atomic number 0
                dummy_indices.append(atom.GetIdx())
        return dummy_indices


    def get_mainchain_atoms(self):
        """
        Finds the set of main chain atoms, determined from all self avoiding paths between two dummy atoms.
    
        Args
        ----
        - self
    
        Returns
        -------
        - Set of main chain atoms.
        
        Example usage
        -------------
        mol = Chem.MolFromSmiles('[*]C1=CC=C(OCCCCCCCCCC)C(C2=CC([*])=CC=C2OCCCCCCCCCC)=C1')
        ts = ToSubstructures(mol)
        highlight_atoms = ts.get_mainchain_atoms()
        Draw.MolToImage(mol, highlightAtoms=list(highlight_atoms), size=(600, 400))
        """
        
        dummies = self.find_dummy_atoms()
        if len(dummies)!=2:
            raise ValueError(f"There are {len(dummies)} dummy atoms. This funtion requires 2.")
        else:
            i, j = dummies[0], dummies[1]
            xss = self.get_self_avoiding_paths(i, j)
            flat = [x for xs in xss for x in xs]
        
        return set(flat)


    def find_non_ring_single_bonds(self):
        """
        Finds all non-aromatic single bonds in a molecule.
        
        Args
        ----
        - self
    
        Returns
        -------
        List of tuples representing the atom indices of non-aromatic single bonds.

        Example usage
        -------------
        mol = Chem.MolFromSmiles('[*]C1=CC=C(OCCCCCCCCCC)C(C2=CC([*])=CC=C2OCCCCCCCCCC)=C1')
        ts = ToSubstructures(mol)
        print(ts.find_non_ring_single_bonds())
        """

        non_aromatic_single_bonds = []
        
        # Iterate through all bonds in the molecule:
        for bond in self.mol.GetBonds():
            # Check if the bond is single and not aromatic:
            if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                # Append the indices of the bonded atoms:
                non_aromatic_single_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        return non_aromatic_single_bonds

    

    def bonds_to_be_broken(self, break_mainchain_bonds=True, break_sidechain_bonds=False):
        """
        Finds the atom indices at the break points defining the resulting fragments. 
        Similar feature to list(BRICS.FindBRICSBonds(mol)) but more specific to this class functionality.
        
        Args
        ----
        - self
        - break_mainchain_bonds : Bool - True returns bonds in the main chain that will be broken.
        - break_sidechain_bonds : Bool - True returns bonds in the side chain that will be broken.
        
    
        Returns
        -------
        List of tuples representing the atom indices of the bonds to be broken

        Example usage
        -------------
        mol = Chem.MolFromSmiles('[*]C1=CC=C(OCCCCCCCCCC)C(C2=CC([*])=CC=C2OCCCCCCCCCC)=C1')
        ts = ToSubstructures(mol)
        bonds = ts.bonds_to_be_broken(break_mainchain_bonds=True, break_sidechain_bonds=True)
        print(bonds)
        """
        
        self.mc_atoms = self.get_mainchain_atoms()
        res = self.find_non_ring_single_bonds()
        bonds=[]
        for bond in res:
            # print(re)
            a, b = bond
            atom_a = self.mol.GetAtomWithIdx(a).GetAtomicNum()
            atom_b = self.mol.GetAtomWithIdx(b).GetAtomicNum()
            # if both atoms are in the main chain
            if self._break_cond(a, b,
                          break_mainchain_bonds=break_mainchain_bonds, 
                          break_sidechain_bonds=break_sidechain_bonds) and not (atom_a== 0 or atom_b == 0):
                
                bonds.append(bond)
        return bonds



    def _break_cond(self, a,b, break_mainchain_bonds=True, break_sidechain_bonds=False):
        # Helper function
        # when break_mainchain_bonds=False and break_mainchain_bonds=True, this will return true for breaking the side chains only
        if a in self.mc_atoms and b in self.mc_atoms:
            # For main chain breaking want the atoms a AND b to be in mainchain_atoms
            if break_mainchain_bonds:
                return True
            else:
                return False
        elif a in self.mc_atoms or b in self.mc_atoms:
            # for side chain breaking want the atoms a OR b in the mainchain_atoms but not a AND b together.
            if break_sidechain_bonds:
                return True
            else:
                return False
        else:
            return False

        
        
    def break_the_bond(self, a, b, replace_a_with=0, replace_b_with=0) -> None:
        """
        Breaks bond at atom positions and replaces with dummy atoms unless specified otherwise.
        
        Args
        ----
        - self
        - a, b: integer atom numbers a and b in the mol
        - replace_a_with: Integer for input Chem.Atom(replace_a_with) defaults to 0 for dummy atom
        - replace_b_with: Integer for input Chem.Atom(replace_a_with) defaults to 0 for dummy atom
    
        Returns
        -------
        - None, but updates self.em, self.nAts 
    
        Example usage
        -------------
        mol = Chem.MolFromSmiles('[*]C(C=C1)=CC2=C1C(C=CC(C3=CC=C(C4=CC=C(C5=CC=C([*])S5)C6=NSN=C46)S3)=C7)=C7N2C(CCCCCCCC)CCCCCCCC')
        a,b = (14, 15)
        ts = ToSubstructures(mol)
        ts.break_the_bond(a, b, replace_a_with=8, replace_b_with=0) # replaces a with an O and b with * (default)
        p = ts.em.GetMol()
        Chem.SanitizeMol(p)
        fragments = Chem.GetMolFrags(p, asMols=True)
        Draw.MolsToGridImage(fragments)
        """
        self.em.RemoveBond(a,b)
        self.em.AddAtom(Chem.Atom(replace_a_with))
        self.em.AddBond(a,self.nAts,Chem.BondType.SINGLE)
        self.em.AddAtom(Chem.Atom(replace_b_with))
        self.em.AddBond(b,self.nAts+1,Chem.BondType.SINGLE)
        self.nAts+=2
        return None
    
    
    def break_all_bonds(self, bonds):
        """
        Breaks bond at atom positions and replaces with dummy atoms unless specified otherwise.
        
        Args
        ----
        - mol: RDKit Mol object
        - bonds: list of bonds (tuples of atom indices) to be broken
    
        Returns
        -------
        - Tuple of fragment mol objects
    
        Example usage
        -------------
        mol = Chem.MolFromSmiles('[*]C(C=C1)=CC2=C1C(C=CC(C3=CC=C(C4=CC=C(C5=CC=C([*])S5)C6=NSN=C46)S3)=C7)=C7N2C(CCCCCCCC)CCCCCCCC')
        bonds=[(10,11), (14,15), (18,19)]
        ts = ToSubstructures(mol)
        fragments = ts.break_all_bonds(bonds)
        Draw.MolsToGridImage(fragments)
        """

        for a,b in bonds:
            self.break_the_bond(a, b, replace_a_with=0, replace_b_with=0)
        p = self.em.GetMol()
        Chem.SanitizeMol(p)
        self.em_mol = p
        self.fragments = Chem.GetMolFrags(p, asMols=True)
        return self.fragments


    def get_substructures(self, break_mainchain_bonds, break_sidechain_bonds, clean_up_dummies=False):
        """
        Returns substructures of a molecule partitioned at any single bonds in the main chain (when break_mainchain_bonds=True)
        And breaks the bond between the side chain and main chain (when break_sidechain_bonds=True)
        
        Args
        ----
        - mol: RDKit Mol object
        - bonds: list of bonds (tuples of atom indices) to be broken
    
        Returns
        -------
        - Tuple of fragment mol objects
    
        Example usage
        -------------
        smiles = '[*]C(S1)=CC2=C1C(C=C(C(CCCCCCCCCCCCCCCC)(CCCCCCCCCCCCCCCC)C3=C4SC(C5=CC=C([*])C6=NSN=C65)=C3)C4=C7)=C7C2(CCCCCCCCCCCCCCCC)CCCCCCCCCCCCCCCC'
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToImage(mol, size = (600,600))
        ts = ToSubstructures(mol)
        frags = ts.get_substructures(break_mainchain_bonds=True, break_sidechain_bonds=False)
        Draw.MolsToGridImage(frags)
        frag_mol = frags[0]
        ts = ToSubstructures(frag_mol)
        fragsfrags = ts.get_substructures(break_mainchain_bonds=False, break_sidechain_bonds=True, clean_up_dummies=True)
        Draw.MolsToGridImage(fragsfrags, subImgSize=(500, 350))
        """
        bonds = self.bonds_to_be_broken(break_mainchain_bonds, break_sidechain_bonds)
        fragments = self.break_all_bonds(bonds)

        if clean_up_dummies:
            mc_frag = fragments[0] # assuming fragments[0] is always the MC rigid group
            dummies = [atom.GetIdx() for atom in mc_frag.GetAtoms() if atom.GetAtomicNum() == 0] # get list of all dummies
            dummies_to_cap = dummies[2:] # assuming it is always the first two dummies that are the backbone bonds - so leaving these alone
            mc_frag = self._cap_the_dummies(mc_frag, dummies_to_cap)
            fragments = (mc_frag,) + fragments[1:]
        return fragments
    
    def _cap_the_dummies(self, mol, dummies, replace_with=1):
        """
        Helper function that returns the molecule with the specified dummy indices capped with some atom (default: Hydrogen)
        Does not work if dummy atoms are part of a ring
        
        Args
        ----
        - mol: RDKit Mol object
        - dummies: List of integers corresponding to the dummy atoms to be replaced with Hs
        - replace_with: Integer corresponding to the atomic number of the atom replacing the dummy
    
        Returns
        -------
        - RDKit Mol object
    
        Example usage
        -------------
        # smiles, dummies = '*c1cc(*)c(*)cc1*', [6, 4, 9, 0]
        smiles, dummies = '*c1cc2c(*)c3sc(*)cc3c(*)c2s1', [0, 13, 5, 9]
        mol = Chem.MolFromSmiles(smiles)
        ts = ToSubstructures(Chem.MolFromSmiles('C')) # the initialised mol is not relevant for this function
        ts._cap_the_dummies(mol, dummies, replace_with=6) 
        """
        dummies.sort(reverse=True)
        nAts = mol.GetNumAtoms()
        em = Chem.EditableMol(mol)
        for b in dummies:
            atom_b = mol.GetAtomWithIdx(b)
            atom_a = atom_b.GetNeighbors()[0]
            a = atom_a.GetIdx()
            em.RemoveAtom(b)
            em.AddAtom(Chem.Atom(replace_with))
            if b==0:
                em.AddBond(0,nAts-1, Chem.BondType.SINGLE)
            else:
                em.AddBond(a,nAts-1, Chem.BondType.SINGLE)
        new_mol = em.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))





class ClusterMols:
    """
    
    Performs molecular clustering using the fingerprints of molcules
    
    """  
    
    def __init__(self, mols, generator=rdFingerprintGenerator.GetMorganGenerator):
        """
        Initialize class

        Inputs
        ------
        mols : List
            Each item is an RDKit Mol.
        generator : rdkit.Chem.rdFingerprintGenerator.method
            where method is e.g. GetMorganGenerator, GetRDKitFPGenerator, ...
        
        """
        self.mols = mols
        self.generator = generator
        self.fingerprints = None
   
    def set_fps(self, **kwargs):
        """
        Manually set the generator arguments for generating the fingerprints
    
        Inputs
        ------
        kwargs : generator args
            e.g. 'fpSize' and 'radius' for the generator: rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator()
    
        Returns
        -------
        if return_xy=True:
            x, y : Tuple
                x : List of all fingerprint sizes (in powers of 2 e.g. ... 64, 128, 256, 512, 1024, 2048 ...)
                y : Number of nonzero values across all molecules in 'self.mols'
            
        Example
        -------
        import matplotlib.pyplot as plt
        smiles = [
            "[*]C1=C(CCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCCCC)C=C([*])S1", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCC)CCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCC)CCCCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCCCCCC)CCCCCCCCCCCC"
        ]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        cm = ClusterMols(mols)
        cm.set_fps(radius=3, fpSize=1024)
        print(cm.gen_kwargs)
        """
        mfpgen = self.generator(**kwargs)
        self.gen_kwargs = {**kwargs}
        self.fingerprints = [mfpgen.GetFingerprint(mol) for mol in self.mols]
        

    def optimize_fps(self, maxIterations=20, return_xy=False, **kwargs):
        """
        Optimizes the size of the fingerprint bit vector for a given radius.
    
        Inputs
        ------
        max_iterations : Int
            Cutoff to avoid infinite loop
        kwargs : generator args
            e.g. 'radius' for the generator: rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator()
    
        Returns
        -------
        if return_xy=True:
            x, y : Tuple
                x : List of all fingerprint sizes (in powers of 2 e.g. ... 64, 128, 256, 512, 1024, 2048 ...)
                y : Number of nonzero values across all molecules in 'self.mols'
            
        Example
        -------
        import matplotlib.pyplot as plt
        smiles = [
            "[*]C1=C(CCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCCCC)C=C([*])S1", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCC)CCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCC)CCCCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCCCCCC)CCCCCCCCCCCC"
        ]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        cm = ClusterMols(mols)
        x,y = cm.optimize_fps(radius=3, return_xy=True)
        plt.semilogx(x, y)
        plt.axvline(x=cm.gen_kwargs['fpSize'], c='r')
        """
        
        i = 0
        x = []
        y = []

    
        with tqdm(total=maxIterations, desc="Maximizing fingerprint size") as pbar:
            while True:
                size = 2 ** i
                mfpgen = self.generator(fpSize=size, **kwargs)
                FPs = [mfpgen.GetFingerprint(mol) for mol in self.mols]
    
                # Logical OR operation on fingerprints to get the number of unique bits
                unique_bits = np.logical_or.reduce([np.array(fp) for fp in FPs]).sum()
    
                x.append(size)
                y.append(unique_bits)
    
                # Check if the number of unique bits stops increasing
                if i > 0 and y[i] <= y[i - 1]:
                    print("Maximum found. Stopping.")
                    break
    
                i += 1
                pbar.update(1)  # Update the progress bar
    
                # Safety check to avoid infinite loop
                if i >= maxIterations:
                    print("Reached maximum iterations. Stopping.")
                    break

        # self.optFpSize = 

        self.gen_kwargs = {'fpSize': x[-2], **kwargs}
        mfpgen = self.generator(**self.gen_kwargs)
        self.fingerprints = [mfpgen.GetFingerprint(mol) for mol in self.mols]
        print("Updated fingerprints.")
        
        if return_xy:
            return x, y


    def tanimoto_distance_matrix(self, fingerprints):
        """
        Computes the (1 - Tanimoto similarity) for each molecule in self.mols i.e. distance matrix
        
        Inputs
        ------
        fingerprints : List
            Each item is a <class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>

        Returns
        -------
        distance matrix : List
            Containing items of the pairwise dissimilarity of each molecule in order of left to right in lower triangular matrix
        
        Example
        -------
        from rdkit.Chem import Draw
        smiles = [
            "[*]C1=C(CCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCC)C=C([*])S1", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCC)CCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCC)CCCCCCCC", 
        ]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        cm = ClusterMols(mols, generator=rdFingerprintGenerator.GetMorganGenerator)
        cm.maximize_fpSize(radius=10)
        print(cm.tanimoto_distance_matrix(cm.fingerprints))
        Draw.MolsToGridImage(mols, legends=list(np.arange(0,len(mols)).astype(str)))
        """
        dists=[]
        for i in range(len(fingerprints)):
            sim = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
            dists.extend([1-sim_i for sim_i in sim])
        return dists


    def cluster_fingerprints(self, fingerprints, distThresh=0.2, asMols=True, keymols=None):
        """
        Clusters mols for a given list of corresponding fingerprints
        
        Inputs
        ------
        fingerprints : List
            Each item is a <class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>
        distThresh : Float
            Defines the cutoff Tanimoto distance for clustering
        asMols : Bool
            Return the clusters as RDKit Mol objects (nested list)
        Returns
        -------
        - List of molecules as RDKit.Chem.Mol objects
        - Tuple of tuples containing indicies
        
        Example
        -------
        from rdkit.Chem import Draw
        smiles = [
            "[*]C1=C(CCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCC)C=C([*])S1", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCC)CCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCC)CCCCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCCCCCC)CCCCCCCCCCCC"
        ]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        cm = ClusterMols(mols, generator=rdFingerprintGenerator.GetMorganGenerator)
        cm.maximize_fpSize(radius=10)
        clusters = cm.cluster_fingerprints(cm.fingerprints, distThresh=0.5, asMols=True)
        Draw.MolsToGridImage(clusters[1])
        """
        
        if keymols is None:
            keymols=self.mols
        
        distance_matrix = self.tanimoto_distance_matrix(fingerprints)
        clusters = Butina.ClusterData(distance_matrix, nPts=len(fingerprints), distThresh=distThresh, isDistData=True)
        # clusters = sorted(clusters, key=len, reverse=True)
        if asMols:
            clustered_mols=[]
            for i in range(len(clusters)):
                clustered_mols.append(pd.Series(keymols).loc[list(clusters[i])].tolist())

            return clustered_mols
        else:
            return clusters


    def cluster_mols(self, distThresh, keymols=None, maxIterations=20, fingerprints=None, asMols=True):
        """
        Clusters molecules using their fingerprints and a Tanimoto distance threshold.

        Parameters
        ----------
        distThresh : float
            Cutoff Tanimoto distance for clustering.
        keymols : list, optional
            A list of RDKit Mol objects corresponding to the fingerprints. Defaults to `self.mols`.
        maxIterations : int, optional
            Maximum number of iterations for optimizing fingerprint size. Defaults to 20.

        Returns
        -------
        list
            A list of clustered RDKit Mol objects.

        Example
        -------
        smiles = [
            "[*]C1=C(CCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCC)C=C([*])S1", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCC)CCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCC)CCCCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCCCCCC)CCCCCCCCCCCC"
        ]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        cm = ClusterMols(mols)
        cm.maximize_fpSize()
        clustered_mols = cm.cluster_mols(radius=6, distThresh=0.4)
        Draw.MolsToGridImage(clustered_mols[1])
        """
        if fingerprints is None and self.fingerprints is None:
            raise ValueError(
                "No fingerprints provided. Either pass a valid list of fingerprints "
                "or call `ClusterMols.maximize_fpSize` to generate optimized fingerprints."
            )

        return self.cluster_fingerprints(self.fingerprints, distThresh=distThresh, asMols=asMols, keymols=keymols)



    def get_similarities(self, ref_mol, measure=DataStructs.TanimotoSimilarity):

        """
        Example
        -------
        smiles = [
            "[*]C1=C(CCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCC)C=C([*])S1", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCC)CCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCC)CCCCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCCCCCC)CCCCCCCCCCCC"
        ]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        ref_mol = Chem.MolFromSmiles('[*]C1=CC=C([*])S1')
        cm = ClusterMols(mols)
        cm.optimize_fps(radius=3)
        cm.get_similarities(ref_mol)
        """
        cm = ClusterMols([ref_mol])
        cm.optimize_fps()
        if cm.gen_kwargs['fpSize'] > self.gen_kwargs['fpSize']:
            print("Warning: May have some bit clashing in the reference fingerprint")

        
        mfpg = self.generator(**self.gen_kwargs)
        ref_fp = mfpg.GetFingerprint(ref_mol)

        similarities = [measure(ref_fp, self.fingerprints[i]) for i in range(len(self.fingerprints))]
        
        return similarities

    

    def sort_mols_by_similarity(self, ref_mol, keymols=None, measure=DataStructs.TanimotoSimilarity, asMols=True):
        """
        Sorts molecules by similarity to a reference molecule.

        Parameters
        ----------
        ref_mol : RDKit Mol
            The reference molecule for similarity comparisons.
        keymols : list, optional
            A list of RDKit Mol objects corresponding to the fingerprints. Defaults to `self.mols`.
        measure : callable, optional
            Similarity metric, such as `DataStructs.TanimotoSimilarity`. Defaults to Tanimoto similarity.
        asMols : bool, optional
            If True, returns a list of RDKit Mol objects sorted by similarity; otherwise, returns indices. Defaults to True.
        fingerprints : List | None, optional

        Returns
        -------
        list
            If `asMols` is True, returns a list of RDKit Mol objects sorted by similarity. 
            Otherwise, returns a list of indices sorted by similarity.

        Example
        -------
        smiles = [
            "[*]C1=C(CCCCCCCC)C=C([*])S1", 
            "[*]C1=C(CCCCCCCCCC)C=C([*])S1", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCC)CCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCC)CCCCCCCC", 
            "[*]C(C=C1)=CC2=C1C(C=CC([*])=C3)=C3C2(CCCCCCCCCCCC)CCCCCCCCCCCC"
        ]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        ref_mol = Chem.MolFromSmiles('[*]C1=CC=C([*])S1')
        cm = ClusterMols(mols)
        cm.optimize_fps(radius=3)
        sorted_mols = cm.sort_mols_by_similarity(ref_mol)
        Draw.MolsToGridImage(sorted_mols)
        """
        
        if self.fingerprints is None:
            raise ValueError(
                "No fingerprints provided. Either pass a valid list of fingerprints with ClusterMols.set_fps "
                "or call `ClusterMols.optimize_fps` to generate optimized fingerprints."
            )
        similarities = self.get_similarities(ref_mol, measure=measure)
                    
        if keymols is None:
            keymols=self.mols


        mols_with_scores = list(
            zip(
                keymols, similarities, list(range(len(keymols)))
            )
        )
        sorted_mols_with_scores = sorted(mols_with_scores, key=lambda x: x[1], reverse=True)
        if asMols:
            return [mol for mol, _ , _ in sorted_mols_with_scores]   
        else:
            return [idx for _, _, idx in sorted_mols_with_scores]
     






    