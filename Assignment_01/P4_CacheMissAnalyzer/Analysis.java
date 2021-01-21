import java.util.*;
import static java.util.stream.Collectors.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Method;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeProperty;
import org.antlr.v4.runtime.tree.TerminalNode;

// FIXME: You should limit your implementation to this class. You are free to add new auxilliary classes. You do not need to touch the LoopNext.g4 file.
class Analysis extends LoopNestBaseListener {

    // Possible types
	
    enum Types {
        Byte, Short, Int, Long, Char, Float, Double, Boolean, String
    }

    // Type of variable declaration
    enum VariableType {
        Primitive, Array, Literal
    }

    // Types of caches supported
    enum CacheTypes {
        DirectMapped, SetAssociative, FullyAssociative,
    }

    // auxilliary data-structure for converting strings
    // to types, ignoring strings because string is not a
    // valid type for loop bounds
    final Map<String, Types> stringToType = Collections.unmodifiableMap(new HashMap<String, Types>() {
        private static final long serialVersionUID = 1L;

        {
            put("byte", Types.Byte);
            put("short", Types.Short);
            put("int", Types.Int);
            put("long", Types.Long);
            put("char", Types.Char);
            put("float", Types.Float);
            put("double", Types.Double);
            put("boolean", Types.Boolean);
        }
    });

    // auxilliary data-structure for mapping types to their byte-size
    // size x means the actual size is 2^x bytes, again ignoring strings
    final Map<Types, Integer> typeToSize = Collections.unmodifiableMap(new HashMap<Types, Integer>() {
        private static final long serialVersionUID = 1L;

        {
            put(Types.Byte, 0);
            put(Types.Short, 1);
            put(Types.Int, 2);
            put(Types.Long, 3);
            put(Types.Char, 1);
            put(Types.Float, 2);
            put(Types.Double, 3);
            put(Types.Boolean, 0);
        }
    });

    // Map of cache type string to value of CacheTypes
    final Map<String, CacheTypes> stringToCacheType = Collections.unmodifiableMap(new HashMap<String, CacheTypes>() {
        private static final long serialVersionUID = 1L;

        {
            put("FullyAssociative", CacheTypes.FullyAssociative);
            put("SetAssociative", CacheTypes.SetAssociative);
            put("DirectMapped", CacheTypes.DirectMapped);
        }
    });
    
    //variables required for storing program and analysis data, cache model
    CacheModel chModel;
    long currLinesUsed;
	HashMap<String,Types> variableTyp = new HashMap<String,Types>();
	List<String> arrList = new ArrayList<String>();
	HashMap<String,ArrayList> arrayDims = new HashMap<String,ArrayList>();
	HashMap<String,Long> intVarValues = new HashMap<String,Long>();
	HashMap<String,Double> doubleVarValues = new HashMap<String,Double>();
	List<HashMap> missList = new ArrayList<HashMap>();
	Stack<ForLoopModel> flmStack = new Stack<ForLoopModel>();
	HashMap<String,ArrayList> arrayForRef = new HashMap<String,ArrayList>();
	HashMap<String,ArrayList> arrayIndRef = new HashMap<String,ArrayList>();
	

    public Analysis() {
    }

    // FIXME: Feel free to override additional methods from
    // LoopNestBaseListener.java based on your needs.
    // Method entry callback
    @Override
    public void enterMethodDeclaration(LoopNestParser.MethodDeclarationContext ctx) {
    	
    	//Initializing a new cache on entering a method
        System.out.println("enterMethodDeclaration");
        //listMethodsforMethodDeclaration(ctx);
        String methodName = getMethodName(ctx);
        System.out.println("Method Name:" + methodName);
		chModel = new CacheModel();
		
    }
    
    
    public void listMethodsforMethodDeclaration(LoopNestParser.MethodDeclarationContext ctx) {
    	//System.out.println("Classname: " + ctx.getClass().getName());
        Class tClass = ctx.getClass();
		Method[] methods = tClass.getMethods();
		/*for (int i = 0; i < methods.length; i++) {
			System.out.println("public method: " + methods[i]);
		}*/
		
    }
    
    public String getMethodName(LoopNestParser.MethodDeclarationContext ctx) {
    	TerminalNode ctx2 = ctx.methodHeader().methodDeclarator().Identifier();
		Class tClass2 = ctx2.getClass();
		//System.out.println("Identifier class name: " + tClass2.getName());
		/*Method[] methods2 = tClass2.getMethods();
		for (int i = 0; i < methods2.length; i++) {
			System.out.println("public method: " + methods2[i]);
		}*/
		return ctx2.toString();
    }

    // End of testcase
    @Override
    public void exitMethodDeclarator(LoopNestParser.MethodDeclaratorContext ctx) {
        System.out.println("exitMethodDeclarator");
    }
    
	@Override public void enterMethodBody(LoopNestParser.MethodBodyContext ctx) { 
		//System.out.println("Entered method body - Class Name: " + ctx.getClass().getName());
	}

	@Override public void exitMethodBody(LoopNestParser.MethodBodyContext ctx) {
		//Computing cache miss after gathering the required data
		computeCacheMisses();
		
		//resetting the required variables after a method being used
		variableTyp.clear();
		arrayDims.clear();
		intVarValues.clear();
		doubleVarValues.clear();
		flmStack.clear();
		arrayIndRef.clear();
		arrayForRef.clear();
		arrList.clear();
	}
	
	public void computeCacheMisses() {
		
		HashMap<String,Long> cacheMisses = new HashMap<String,Long>();
		
		for(int i=0;i<arrList.size();i++) {
			String arrName = arrList.get(i);
			currLinesUsed=1;
			long miss = computeCacheMissforArray(arrName);
			cacheMisses.put(arrName, miss);
			System.out.println("Array Name: " + arrName + " - Total Miss: " + miss);
		}
		
		missList.add(cacheMisses);
	}
	
	public long computeCacheMissforArray(String arrayName) {
		
		long miss=0;
		List<ForLoopModel> forLoops = arrayForRef.get(arrayName);
		if(forLoops.size()==0) {
			return miss;
		}
		List<String> indexes = arrayIndRef.get(arrayName);
		
		//check miss in inner loop and if array can fit into cache for access over inner loop
		long innerMiss = computeInnerMiss(arrayName,indexes,forLoops.get(forLoops.size()-1));
		//System.out.println(arrayName + " inner loop miss: " + innerMiss);
		boolean canInnerLoopFit = testArrayFitInnerLoop(arrayName,indexes,forLoops.get(forLoops.size()-1),innerMiss);
		//System.out.println(arrayName + " inner loop access can Fit: " + canInnerLoopFit);
		
		int forLoopLen = forLoops.size();
		
		for(int i=forLoopLen-2;i>=0;i--) {
			
			//over each loop check total miss including that and inner loop depending whether inner loop can fit or not
			innerMiss = computeLoopMiss(arrayName,i,indexes,forLoops,innerMiss,canInnerLoopFit);
			//over each loop check if array can fit for access including that and inner loop
			canInnerLoopFit = computeArrayFit(arrayName,i,indexes,forLoops,innerMiss,canInnerLoopFit);
			//System.out.println(arrayName + " miss upto loop " + forLoops.get(i).getInitVar() + ":" + innerMiss);
			//System.out.println(arrayName + " access can fit upto loop " + forLoops.get(i).getInitVar() + ":" +canInnerLoopFit);
		}
				
		return innerMiss;
	}
	
	public long computeLoopMiss(String arrayName,int ind,List<String> indexes,List<ForLoopModel> FlmList,long innerMiss,boolean canInnerLoopFit) {
		
		long missFactor=1;
		
		ForLoopModel currFlm = FlmList.get(ind);
		long numberAccess = Math.max((long)0,(long)((currFlm.getUpperBound() - currFlm.getInitValue() - (long)1)/(long)currFlm.getStride()) + (long)1);
		if(numberAccess==0) {
			return 0;
		}
		ArrayList<Long> dimList = arrayDims.get(arrayName);
		
		String currvar = currFlm.getInitVar();
		long multCoeff = 1;
		long dist=0;
		Types arrType = variableTyp.get(arrayName);
		long typeSize = typeToSize.get(arrType);
		typeSize = (long)1<<typeSize;
		long wordsPerBlock = ((long)1<<chModel.getBlockPower())/typeSize;
		long wordsPerCache = ((long)1<<chModel.getCachePower())/typeSize;
		
		for(int i=dimList.size()-1;i>=0;i--) {
			String arrInd = indexes.get(i);
			if(arrInd.equals(currvar)) {
				dist+= (currFlm.getStride())*multCoeff;
			}
			multCoeff = multCoeff*dimList.get(i);
		}
		
		if(canInnerLoopFit==false) {
			return innerMiss*numberAccess;
		}
		else {
			if(dist==0) {
				return innerMiss;
			}
			else {
				if(dist<wordsPerBlock) {
					missFactor = (long)Math.ceil((double)(dist*(numberAccess-1) + 1)/(double)(wordsPerBlock));
					return missFactor*innerMiss;
				}
				else {
					return numberAccess*innerMiss;
				}
			}
		}
		
	}
	
	public boolean computeArrayFit(String arrayName,int ind,List<String> indexes,List<ForLoopModel> FlmList,long innerMiss,boolean canInnerLoopFit) {
		
		if(canInnerLoopFit==false) {
			return false;
		}
		
		ForLoopModel currFlm = FlmList.get(ind);
		long numberAccess = Math.max((long)0,(long)((currFlm.getUpperBound() - currFlm.getInitValue() - (long)1)/(long)currFlm.getStride()) + (long)1);
		ArrayList<Long> dimList = arrayDims.get(arrayName);
		
		String currvar = currFlm.getInitVar();
		long multCoeff = 1;
		long dist=0;
		Types arrType = variableTyp.get(arrayName);
		long typeSize = typeToSize.get(arrType);
		typeSize = (long)1<<typeSize;
		long wordsPerBlock = ((long)1<<chModel.getBlockPower())/typeSize;
		long wordsPerCache = ((long)1<<chModel.getCachePower())/typeSize;
		
		for(int i=dimList.size()-1;i>=0;i--) {
			String arrInd = indexes.get(i);
			if(arrInd.equals(currvar)) {
				dist+= (currFlm.getStride())*multCoeff;
			}
			multCoeff = multCoeff*dimList.get(i);
		}
		
		if(dist==0) {
			return true;
		}
		else {
			if(chModel.getCacheType().equals("FullyAssociative")) {
				long numBlocks = wordsPerCache/wordsPerBlock;
				return (innerMiss<=numBlocks);
			}
			else {
				long linesPerSet = chModel.getSetSize();
				long numBlocksCache = wordsPerCache/wordsPerBlock;
				long numSet = numBlocksCache/linesPerSet;
				long numBbits = log2(wordsPerBlock);
				long numSbits = log2(numSet);
				long combSBbits = ((long)1<<(numBbits + numSbits));
				
				/*long finalElementDist = (numberAccess-1)*dist;
				long iterBits = Math.max(combSBbits, dist);
				
				long clu = finalElementDist/iterBits + 1;
				currLinesUsed = currLinesUsed*clu;*/
				
				multCoeff = 1;
				
				for(int i=indexes.size()-1;i>=0;i--) {
					String arrInd = indexes.get(i);
					
					if(arrInd.equals(currvar)) {
						long indDist = currFlm.getStride()*multCoeff;
						long iterBits = Math.max(indDist, combSBbits);
						long indFinEleDist = (numberAccess-1)*currFlm.getStride()*multCoeff;
						
						long clu = indFinEleDist/iterBits + 1;
						currLinesUsed = currLinesUsed*clu;
					}
					
					multCoeff*=dimList.get(i);
				}
				
				if(currLinesUsed<=linesPerSet) {
					return true;
				}
				else {
					return false;
				}
				
			}
		}
		
	}
	
	public boolean testArrayFitInnerLoop(String arrayName, List<String> indexes,ForLoopModel innerFlm,long innerMiss) {
		boolean b = true;
		long numberAccess = Math.max((long)0,(long)((innerFlm.getUpperBound() - innerFlm.getInitValue() - (long)1)/(long)innerFlm.getStride()) + (long)1);
		
		if(numberAccess<=1) {
			return true;
		}
		
		Types arrType = variableTyp.get(arrayName);
		long typeSize = typeToSize.get(arrType);
		typeSize = (long)1<<typeSize;
		long wordsPerBlock = ((long)1<<chModel.getBlockPower())/typeSize;
		long wordsPerCache = ((long)1<<chModel.getCachePower())/typeSize;
		
		if(chModel.cacheType.equals("FullyAssociative")) {
			long numBlocks = wordsPerCache/wordsPerBlock;
			if(numBlocks>=innerMiss) {
				return true;
			}
			return false;
		}
		else {
			long linePerSet = chModel.getSetSize();
			long numSets = (wordsPerCache)/(wordsPerBlock * linePerSet); 
			long numBbits = log2(wordsPerBlock);
			long numSbits = log2(numSets);
			
			ArrayList<Long> dimList = arrayDims.get(arrayName);
			String currvar = innerFlm.getInitVar();
			long dist=0, multCoeff = 1;
			for(int i=dimList.size()-1;i>=0;i--) {
				String arrInd = indexes.get(i);
				if(arrInd.equals(currvar)) {
					dist+= (innerFlm.getStride())*multCoeff;
				}
				multCoeff = multCoeff*dimList.get(i);
			}
			
			long combSBbits = ((long)1<<(numBbits + numSbits));
			
			/*long finalElementDist = (numberAccess-1)*dist;
			long iterBits = Math.max(combSBbits, dist);
			
			long clu = finalElementDist/iterBits + 1;
			currLinesUsed = currLinesUsed*clu;*/
			
			multCoeff = 1;
			
			for(int i=indexes.size()-1;i>=0;i--) {
				String arrInd = indexes.get(i);
				
				if(arrInd.equals(currvar)) {
					long indDist = innerFlm.getStride()*multCoeff;
					long iterBits = Math.max(indDist, combSBbits);
					long indFinEleDist = (numberAccess-1)*innerFlm.getStride()*multCoeff;
					
					long clu = indFinEleDist/iterBits + 1;
					currLinesUsed = currLinesUsed*clu;
				}
				
				multCoeff*=dimList.get(i);
			}
			
			if(currLinesUsed<=linePerSet) {
				return true;
			}
			else {
				return false;
			}
			
		}

	}
	
	long log2(long x) {
		//log2 method for perfect 2 power numbers
		long log = 0;
		while(x>1) {
			x = x>>1;
			log++;
		}
		return log;
	}
	
	public long computeInnerMiss(String arrayName,List<String> indexes,ForLoopModel innerFlm) {
		long miss = 0;
		long numberAccess = Math.max((long)0,(long)((innerFlm.getUpperBound() - innerFlm.getInitValue() - (long)1)/(long)innerFlm.getStride()) + (long)1);
		if(numberAccess==0) {
			return 0;
		}
		long blockP = chModel.getBlockPower();
		long blockSize = ((long)1<<blockP);
		Types arrtp = variableTyp.get(arrayName);
		long elePerBlock = blockSize/((long)1<<(long)typeToSize.get(arrtp));
		
		String currvar = innerFlm.getInitVar();
		ArrayList<Long> dimList = arrayDims.get(arrayName);
		
		long multCoeff = 1;
		long dist=0;
		for(int i=dimList.size()-1;i>=0;i--) {
			String arrInd = indexes.get(i);
			if(arrInd.equals(currvar)) {
				dist+= (innerFlm.getStride())*multCoeff;
			}
			multCoeff = multCoeff*dimList.get(i);
		}
		
		if(dist>=elePerBlock || numberAccess==0) {
			miss = numberAccess;	//for inner loop, misses will be number of access if subsequent elements cannot fit into a block
		}
		else {
			miss = (long)Math.ceil((double)(dist*(numberAccess-1) + 1)/(double)(elePerBlock));   //misses will be equal to number of blocks fetched
		}
		
		return miss;
	}

    @Override
    public void exitTests(LoopNestParser.TestsContext ctx) {
        try {
            FileOutputStream fos = new FileOutputStream("Results.obj");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            // FIXME: Serialize your data to a file
            oos.writeObject(missList);
            oos.close();
            
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    @Override
    public void exitLocalVariableDeclaration(LoopNestParser.LocalVariableDeclarationContext ctx) {

    }
    
    @Override
	public void enterArrayAccess(LoopNestParser.ArrayAccessContext ctx) {
		
    	insertIntoArrayIndMap(ctx);
    	insertIntoArrayForMap(ctx);
		
	}
    
    public void insertIntoArrayIndMap(LoopNestParser.ArrayAccessContext ctx) {
    	List exprList  = ctx.expressionName();
		String varName = ctx.expressionName(0).Identifier().getText();
		ArrayList<String> indexList = new ArrayList<>();
		
		for(int i=1;i<exprList.size();i++) {
			String indName = ctx.expressionName(i).Identifier().getText();
			indexList.add(indName);
		}
		
		arrayIndRef.put(varName, indexList);
    }
    
    public void insertIntoArrayForMap(LoopNestParser.ArrayAccessContext ctx) {
    	String varName = ctx.expressionName(0).Identifier().getText();
		ArrayList<ForLoopModel> indexList = new ArrayList<ForLoopModel>();
		
		int len = flmStack.size();
		for(int i=0;i<len;i++) {
			ForLoopModel flm = flmStack.get(i);
			indexList.add(flm);
		}
		
		arrayForRef.put(varName, indexList);
		
    }

    @Override
	public void exitArrayAccess(LoopNestParser.ArrayAccessContext ctx) {
		
	}
    
	@Override public void enterArrayAccess_lfno_primary(LoopNestParser.ArrayAccess_lfno_primaryContext ctx) {
		insertIntoArrayIndMap(ctx);
    	insertIntoArrayForMap(ctx);
	}
	
	public void insertIntoArrayIndMap(LoopNestParser.ArrayAccess_lfno_primaryContext ctx) {
		
		//records index variables for array access
    	List exprList  = ctx.expressionName();
		String varName = ctx.expressionName(0).Identifier().getText();
		ArrayList<String> indexList = new ArrayList<>();
		
		for(int i=1;i<exprList.size();i++) {
			String indName = ctx.expressionName(i).Identifier().getText();
			indexList.add(indName);
		}
		
		arrayIndRef.put(varName, indexList);
    }
    
    public void insertIntoArrayForMap(LoopNestParser.ArrayAccess_lfno_primaryContext ctx) {
    	
    	//records hierarchy of loop inside which array is accessed
    	String varName = ctx.expressionName(0).Identifier().getText();
		ArrayList<ForLoopModel> indexList = new ArrayList<ForLoopModel>();
		
		int len = flmStack.size();
		for(int i=0;i<len;i++) {
			ForLoopModel flm = flmStack.get(i);
			indexList.add(flm);
		}
		
		arrayForRef.put(varName, indexList);
		
    }

	@Override public void exitArrayAccess_lfno_primary(LoopNestParser.ArrayAccess_lfno_primaryContext ctx) {
		
	}
    
    @Override
    public void enterVariableDeclarator(LoopNestParser.VariableDeclaratorContext ctx) {
    	String varName = getVariableName(ctx);
    	if(isCacheConfigVariable(varName)) {
    		setupCacheConfig(varName,ctx);
    	}
    	else {
    		String varType = setupVariableType(varName,ctx);
    		variableTyp.put(varName, stringToType.get(varType));
    	}
    }
    
    public String setupVariableType(String varName,LoopNestParser.VariableDeclaratorContext ctx) {
    	
    	//method to store variable and necessary parameters like array dimension
    	LoopNestParser.LocalVariableDeclarationContext parlvdc = (LoopNestParser.LocalVariableDeclarationContext) ctx.getParent();
    	String vtyp = null;
    	if(parlvdc.unannType()!=null) {
    		LoopNestParser.UnannTypeContext uatc = parlvdc.unannType();
    		if(uatc.unannPrimitiveType()!=null) {
    			LoopNestParser.UnannPrimitiveTypeContext uptc = uatc.unannPrimitiveType();
    			vtyp = uptc.getText();
    			vtyp.replace("'","");
    			//System.out.println("var type for " + varName + ":" + vtyp);
    			if(uptc.numericType()!=null) {
    				if(uptc.numericType().integralType()!=null) {
    					long lv = Long.parseLong(ctx.literal().IntegerLiteral().getText());
    					intVarValues.put(varName, lv);
    				}
    				else {
    					double dv = Double.parseDouble(ctx.literal().FloatingPointLiteral().getText());
    					doubleVarValues.put(varName, dv);
    				}
    			}
    		}
    		else {
    			//code for array type
    			LoopNestParser.UnannPrimitiveTypeContext uptc = uatc.unannArrayType().unannPrimitiveType();
    			vtyp = uptc.getText();
    			vtyp.replace("'","");
    			//System.out.println("var type for " + varName + ":" + vtyp);
    			LoopNestParser.DimExprsContext dimextc = ctx.arrayCreationExpression().dimExprs();
    			List dimobs = dimextc.dimExpr();
    			int lendim = dimobs.size();
    			ArrayList dimList = new ArrayList<Long>(lendim);
    			for(int i=0;i<lendim;i++) {
    				LoopNestParser.DimExprContext deob = (LoopNestParser.DimExprContext) dimobs.get(i);
    				long dimval = 0;
    				if(deob.expressionName()!=null) {
    					dimval = intVarValues.get(deob.expressionName().getText());
    				}
    				else if(deob.IntegerLiteral()!=null) {
    					dimval = Long.parseLong(deob.IntegerLiteral().getText());
    				}
    				dimList.add(dimval);
    			}
    			arrList.add(varName);
    			arrayDims.put(varName, dimList);
    			
    		}
    	}
    	else {
    		vtyp = "String";
    	}
    	
    	return vtyp;
    }
    
    public String getVariableName(LoopNestParser.VariableDeclaratorContext ctx) {
    	String varName = ctx.variableDeclaratorId().getText();
    	//System.out.println("variable=" + varName);
    	return varName;
    }
    
    public boolean isCacheConfigVariable(String varName) {
    	String[] variables = {"cachePower","blockPower","setSize","cacheType"};
    	for(int i=0;i<variables.length;i++) {
    		if(variables[i].equals(varName)) {
    			return true;
    		}
    	}
    	
    	return false;

    }
    
    public void setupCacheConfig(String varName, LoopNestParser.VariableDeclaratorContext ctx) {
    	
    	switch(varName) {
    		case "cachePower":
    			chModel.setCachePower(Integer.parseInt(ctx.literal().IntegerLiteral().getText()));
    			break;
    		case "blockPower":
    			chModel.setBlockPower(Integer.parseInt(ctx.literal().IntegerLiteral().getText()));
    			break;
    		case "setSize":
    			chModel.setSetSize(Integer.parseInt(ctx.literal().IntegerLiteral().getText()));
    			break;
    		case "cacheType":
    			String cType = ctx.literal().StringLiteral().getText();
    			cType = cType.substring(1, cType.length()-1);
    			//System.out.println("Cache Type:" + cType);
    			chModel.setCacheType(cType);
    			break;
    	}
    }
    
	@Override 
	public void enterForStatement(LoopNestParser.ForStatementContext ctx) { 
		ForLoopModel flm = createForLoopModel(ctx);
		flmStack.push(flm);
	}
	
	public ForLoopModel createForLoopModel(LoopNestParser.ForStatementContext ctx) {
		ForLoopModel flm = new ForLoopModel();
		setInitVariables(flm,ctx.forInit());
		setUpperBound(flm,ctx.forCondition());
		setStrVal(flm,ctx.forUpdate());
		return flm;
	}
	
	public void setInitVariables(ForLoopModel flm,LoopNestParser.ForInitContext ctx) {
		LoopNestParser.VariableDeclaratorContext vdc = ctx.localVariableDeclaration().variableDeclarator();
		String varName = getVariableName(vdc);
		flm.setInitVar(varName);
		LoopNestParser.LiteralContext lctx = vdc.literal();
		if(lctx.IntegerLiteral()!=null) {
			long initV = Long.parseLong(lctx.IntegerLiteral().getText());
			flm.setInitValue(initV);
		}
		
	}
	
	public void setUpperBound(ForLoopModel flm,LoopNestParser.ForConditionContext ctx) {
		LoopNestParser.RelationalExpressionContext rectx = ctx.relationalExpression();
		long upb = 0;
		if(rectx.expressionName(1)!=null) {
			LoopNestParser.ExpressionNameContext enctx = rectx.expressionName(1);
			String vnam = enctx.Identifier().getText();
			upb = intVarValues.get(vnam);
		}
		else if(rectx.IntegerLiteral()!=null) {
			upb = Long.parseLong(rectx.IntegerLiteral().getText());
		}
		flm.setUpperBound(upb);
	}
	
	public void setStrVal(ForLoopModel flm, LoopNestParser.ForUpdateContext ctx) {
		LoopNestParser.SimplifiedAssignmentContext simctx = ctx.simplifiedAssignment();
		long strd = 0;
		if(simctx.expressionName(1)!=null) {
			LoopNestParser.ExpressionNameContext enctx = simctx.expressionName(1);
			String vnam = enctx.Identifier().getText();
			strd = intVarValues.get(vnam);
		}
		else if(simctx.IntegerLiteral()!=null) {
			strd = Long.parseLong(simctx.IntegerLiteral().getText());
		}
		flm.setStride(strd);
	}
	
	@Override public void exitForStatement(LoopNestParser.ForStatementContext ctx) { 
		flmStack.pop();
	}

    @Override
    public void exitVariableDeclarator(LoopNestParser.VariableDeclaratorContext ctx) {
    }

    @Override
    public void exitArrayCreationExpression(LoopNestParser.ArrayCreationExpressionContext ctx) {
    }

    @Override
    public void exitDimExprs(LoopNestParser.DimExprsContext ctx) {
    }

    @Override
    public void exitDimExpr(LoopNestParser.DimExprContext ctx) {
    }

    @Override
    public void exitLiteral(LoopNestParser.LiteralContext ctx) {
    }

    @Override
    public void exitVariableDeclaratorId(LoopNestParser.VariableDeclaratorIdContext ctx) {
    }

    @Override
    public void exitUnannArrayType(LoopNestParser.UnannArrayTypeContext ctx) {
    }

    @Override
    public void enterDims(LoopNestParser.DimsContext ctx) {
    }

    @Override
    public void exitUnannPrimitiveType(LoopNestParser.UnannPrimitiveTypeContext ctx) {
    }

    @Override
    public void exitNumericType(LoopNestParser.NumericTypeContext ctx) {
    }

    @Override
	public void enterIntegralType(LoopNestParser.IntegralTypeContext ctx) {
    }
    
    @Override
    public void exitIntegralType(LoopNestParser.IntegralTypeContext ctx) {
    }

    @Override
    public void exitFloatingPointType(LoopNestParser.FloatingPointTypeContext ctx) {
    }

    @Override
    public void exitForInit(LoopNestParser.ForInitContext ctx) {
    	
    }

    @Override
    public void exitForCondition(LoopNestParser.ForConditionContext ctx) {
    }

    @Override
    public void exitRelationalExpression(LoopNestParser.RelationalExpressionContext ctx) {
    }

    @Override
    public void exitForUpdate(LoopNestParser.ForUpdateContext ctx) {
    }

    @Override
    public void exitSimplifiedAssignment(LoopNestParser.SimplifiedAssignmentContext ctx) {
    }

    
    //inner class to encapsulate cache information
    public class CacheModel{
    	
    	long cachePower;
    	long blockPower;
    	String cacheType;
    	long setSize;
    	
		public long getCachePower() {
			return cachePower;
		}
		public void setCachePower(long cachePower) {
			this.cachePower = cachePower;
		}
		public long getBlockPower() {
			return blockPower;
		}
		public void setBlockPower(long blockPower) {
			this.blockPower = blockPower;
		}
		public String getCacheType() {
			return cacheType;
		}
		public void setCacheType(String cacheType) {
			this.cacheType = cacheType;
			if(cacheType.equals("DirectMapped")){
				this.setSetSize(1);
			}
		}
		public long getSetSize() {
			return setSize;
		}
		public void setSetSize(long setSize) {
			this.setSize = setSize;
		}
    	
    }
    
    //inner class to encapsulate loop with the considered syntax
    public class ForLoopModel{
    	String initVar;
    	long initValue;
    	long upperBound;
    	long stride;
    	
		public String getInitVar() {
			return initVar;
		}
		public void setInitVar(String initVar) {
			this.initVar = initVar;
		}
		public long getInitValue() {
			return initValue;
		}
		public void setInitValue(long initValue) {
			this.initValue = initValue;
		}
		public long getUpperBound() {
			return upperBound;
		}
		public void setUpperBound(long upperBound) {
			this.upperBound = upperBound;
		}
		public long getStride() {
			return stride;
		}
		public void setStride(long stride) {
			this.stride = stride;
		}
    	
    	
    }

}
