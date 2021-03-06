??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
?
Conv2D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameConv2D_1/kernel
{
#Conv2D_1/kernel/Read/ReadVariableOpReadVariableOpConv2D_1/kernel*&
_output_shapes
: *
dtype0
r
Conv2D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv2D_1/bias
k
!Conv2D_1/bias/Read/ReadVariableOpReadVariableOpConv2D_1/bias*
_output_shapes
: *
dtype0
?
Conv2D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameConv2D_2/kernel
{
#Conv2D_2/kernel/Read/ReadVariableOpReadVariableOpConv2D_2/kernel*&
_output_shapes
: @*
dtype0
r
Conv2D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv2D_2/bias
k
!Conv2D_2/bias/Read/ReadVariableOpReadVariableOpConv2D_2/bias*
_output_shapes
:@*
dtype0
?
Conv2D_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameConv2D_3/kernel
|
#Conv2D_3/kernel/Read/ReadVariableOpReadVariableOpConv2D_3/kernel*'
_output_shapes
:@?*
dtype0
s
Conv2D_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameConv2D_3/bias
l
!Conv2D_3/bias/Read/ReadVariableOpReadVariableOpConv2D_3/bias*
_output_shapes	
:?*
dtype0
?
DeConv2D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*"
shared_nameDeConv2D_1/kernel
?
%DeConv2D_1/kernel/Read/ReadVariableOpReadVariableOpDeConv2D_1/kernel*(
_output_shapes
:??*
dtype0
w
DeConv2D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameDeConv2D_1/bias
p
#DeConv2D_1/bias/Read/ReadVariableOpReadVariableOpDeConv2D_1/bias*
_output_shapes	
:?*
dtype0
?
De2DTrans_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*#
shared_nameDe2DTrans_2/kernel
?
&De2DTrans_2/kernel/Read/ReadVariableOpReadVariableOpDe2DTrans_2/kernel*'
_output_shapes
:@?*
dtype0
x
De2DTrans_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameDe2DTrans_2/bias
q
$De2DTrans_2/bias/Read/ReadVariableOpReadVariableOpDe2DTrans_2/bias*
_output_shapes
:@*
dtype0
?
De2DTrans_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameDe2DTrans_3/kernel
?
&De2DTrans_3/kernel/Read/ReadVariableOpReadVariableOpDe2DTrans_3/kernel*&
_output_shapes
: @*
dtype0
x
De2DTrans_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameDe2DTrans_3/bias
q
$De2DTrans_3/bias/Read/ReadVariableOpReadVariableOpDe2DTrans_3/bias*
_output_shapes
: *
dtype0
?
Conv2DTrans_recon/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameConv2DTrans_recon/kernel
?
,Conv2DTrans_recon/kernel/Read/ReadVariableOpReadVariableOpConv2DTrans_recon/kernel*&
_output_shapes
: *
dtype0
?
Conv2DTrans_recon/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameConv2DTrans_recon/bias
}
*Conv2DTrans_recon/bias/Read/ReadVariableOpReadVariableOpConv2DTrans_recon/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/Conv2D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Conv2D_1/kernel/m
?
*Adam/Conv2D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D_1/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/Conv2D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv2D_1/bias/m
y
(Adam/Conv2D_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D_1/bias/m*
_output_shapes
: *
dtype0
?
Adam/Conv2D_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/Conv2D_2/kernel/m
?
*Adam/Conv2D_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D_2/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/Conv2D_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Conv2D_2/bias/m
y
(Adam/Conv2D_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/Conv2D_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/Conv2D_3/kernel/m
?
*Adam/Conv2D_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D_3/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/Conv2D_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/Conv2D_3/bias/m
z
(Adam/Conv2D_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D_3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/DeConv2D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameAdam/DeConv2D_1/kernel/m
?
,Adam/DeConv2D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/DeConv2D_1/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/DeConv2D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/DeConv2D_1/bias/m
~
*Adam/DeConv2D_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/DeConv2D_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/De2DTrans_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameAdam/De2DTrans_2/kernel/m
?
-Adam/De2DTrans_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/De2DTrans_2/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/De2DTrans_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/De2DTrans_2/bias/m

+Adam/De2DTrans_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/De2DTrans_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/De2DTrans_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/De2DTrans_3/kernel/m
?
-Adam/De2DTrans_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/De2DTrans_3/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/De2DTrans_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/De2DTrans_3/bias/m

+Adam/De2DTrans_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/De2DTrans_3/bias/m*
_output_shapes
: *
dtype0
?
Adam/Conv2DTrans_recon/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/Conv2DTrans_recon/kernel/m
?
3Adam/Conv2DTrans_recon/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2DTrans_recon/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/Conv2DTrans_recon/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/Conv2DTrans_recon/bias/m
?
1Adam/Conv2DTrans_recon/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv2DTrans_recon/bias/m*
_output_shapes
:*
dtype0
?
Adam/Conv2D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Conv2D_1/kernel/v
?
*Adam/Conv2D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D_1/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/Conv2D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv2D_1/bias/v
y
(Adam/Conv2D_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D_1/bias/v*
_output_shapes
: *
dtype0
?
Adam/Conv2D_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/Conv2D_2/kernel/v
?
*Adam/Conv2D_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D_2/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/Conv2D_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Conv2D_2/bias/v
y
(Adam/Conv2D_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/Conv2D_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/Conv2D_3/kernel/v
?
*Adam/Conv2D_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D_3/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/Conv2D_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/Conv2D_3/bias/v
z
(Adam/Conv2D_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D_3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/DeConv2D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameAdam/DeConv2D_1/kernel/v
?
,Adam/DeConv2D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/DeConv2D_1/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/DeConv2D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/DeConv2D_1/bias/v
~
*Adam/DeConv2D_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/DeConv2D_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/De2DTrans_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameAdam/De2DTrans_2/kernel/v
?
-Adam/De2DTrans_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/De2DTrans_2/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/De2DTrans_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/De2DTrans_2/bias/v

+Adam/De2DTrans_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/De2DTrans_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/De2DTrans_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/De2DTrans_3/kernel/v
?
-Adam/De2DTrans_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/De2DTrans_3/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/De2DTrans_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/De2DTrans_3/bias/v

+Adam/De2DTrans_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/De2DTrans_3/bias/v*
_output_shapes
: *
dtype0
?
Adam/Conv2DTrans_recon/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/Conv2DTrans_recon/kernel/v
?
3Adam/Conv2DTrans_recon/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2DTrans_recon/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/Conv2DTrans_recon/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/Conv2DTrans_recon/bias/v
?
1Adam/Conv2DTrans_recon/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv2DTrans_recon/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?W
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?W
value?WB?W B?W
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
R
.regularization_losses
/trainable_variables
0	variables
1	keras_api
R
2regularization_losses
3trainable_variables
4	variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
R
<regularization_losses
=trainable_variables
>	variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
R
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
?
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_ratem?m?m?m?(m?)m?6m?7m?@m?Am?Jm?Km?Pm?Qm?v?v?v?v?(v?)v?6v?7v?@v?Av?Jv?Kv?Pv?Qv?
 
*
@0
A1
J2
K3
P4
Q5
f
0
1
2
3
(4
)5
66
77
@8
A9
J10
K11
P12
Q13
?
[layer_metrics
regularization_losses
trainable_variables

\layers
	variables
]non_trainable_variables
^layer_regularization_losses
_metrics
 
[Y
VARIABLE_VALUEConv2D_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv2D_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
?
`layer_metrics
regularization_losses
trainable_variables

alayers
	variables
bnon_trainable_variables
clayer_regularization_losses
dmetrics
 
 
 
?
elayer_metrics
regularization_losses
trainable_variables

flayers
	variables
gnon_trainable_variables
hlayer_regularization_losses
imetrics
[Y
VARIABLE_VALUEConv2D_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv2D_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
?
jlayer_metrics
 regularization_losses
!trainable_variables

klayers
"	variables
lnon_trainable_variables
mlayer_regularization_losses
nmetrics
 
 
 
?
olayer_metrics
$regularization_losses
%trainable_variables

players
&	variables
qnon_trainable_variables
rlayer_regularization_losses
smetrics
[Y
VARIABLE_VALUEConv2D_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv2D_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

(0
)1
?
tlayer_metrics
*regularization_losses
+trainable_variables

ulayers
,	variables
vnon_trainable_variables
wlayer_regularization_losses
xmetrics
 
 
 
?
ylayer_metrics
.regularization_losses
/trainable_variables

zlayers
0	variables
{non_trainable_variables
|layer_regularization_losses
}metrics
 
 
 
?
~layer_metrics
2regularization_losses
3trainable_variables

layers
4	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
][
VARIABLE_VALUEDeConv2D_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEDeConv2D_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

60
71
?
?layer_metrics
8regularization_losses
9trainable_variables
?layers
:	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
 
 
 
?
?layer_metrics
<regularization_losses
=trainable_variables
?layers
>	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
^\
VARIABLE_VALUEDe2DTrans_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEDe2DTrans_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
?
?layer_metrics
Bregularization_losses
Ctrainable_variables
?layers
D	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
 
 
 
?
?layer_metrics
Fregularization_losses
Gtrainable_variables
?layers
H	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
^\
VARIABLE_VALUEDe2DTrans_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEDe2DTrans_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
?
?layer_metrics
Lregularization_losses
Mtrainable_variables
?layers
N	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
db
VARIABLE_VALUEConv2DTrans_recon/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEConv2DTrans_recon/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
?
?layer_metrics
Rregularization_losses
Strainable_variables
?layers
T	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
8
0
1
2
3
(4
)5
66
77
 

?0
?1
 
 

0
1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

(0
)1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

60
71
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/Conv2D_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv2D_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv2D_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv2D_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/DeConv2D_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/DeConv2D_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/De2DTrans_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/De2DTrans_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/De2DTrans_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/De2DTrans_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Conv2DTrans_recon/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Conv2DTrans_recon/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv2D_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv2D_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv2D_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/DeConv2D_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/DeConv2D_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/De2DTrans_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/De2DTrans_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/De2DTrans_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/De2DTrans_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Conv2DTrans_recon/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Conv2DTrans_recon/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_Conv2D_1_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_Conv2D_1_inputConv2D_1/kernelConv2D_1/biasConv2D_2/kernelConv2D_2/biasConv2D_3/kernelConv2D_3/biasDeConv2D_1/kernelDeConv2D_1/biasDe2DTrans_2/kernelDe2DTrans_2/biasDe2DTrans_3/kernelDe2DTrans_3/biasConv2DTrans_recon/kernelConv2DTrans_recon/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_33977
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv2D_1/kernel/Read/ReadVariableOp!Conv2D_1/bias/Read/ReadVariableOp#Conv2D_2/kernel/Read/ReadVariableOp!Conv2D_2/bias/Read/ReadVariableOp#Conv2D_3/kernel/Read/ReadVariableOp!Conv2D_3/bias/Read/ReadVariableOp%DeConv2D_1/kernel/Read/ReadVariableOp#DeConv2D_1/bias/Read/ReadVariableOp&De2DTrans_2/kernel/Read/ReadVariableOp$De2DTrans_2/bias/Read/ReadVariableOp&De2DTrans_3/kernel/Read/ReadVariableOp$De2DTrans_3/bias/Read/ReadVariableOp,Conv2DTrans_recon/kernel/Read/ReadVariableOp*Conv2DTrans_recon/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Conv2D_1/kernel/m/Read/ReadVariableOp(Adam/Conv2D_1/bias/m/Read/ReadVariableOp*Adam/Conv2D_2/kernel/m/Read/ReadVariableOp(Adam/Conv2D_2/bias/m/Read/ReadVariableOp*Adam/Conv2D_3/kernel/m/Read/ReadVariableOp(Adam/Conv2D_3/bias/m/Read/ReadVariableOp,Adam/DeConv2D_1/kernel/m/Read/ReadVariableOp*Adam/DeConv2D_1/bias/m/Read/ReadVariableOp-Adam/De2DTrans_2/kernel/m/Read/ReadVariableOp+Adam/De2DTrans_2/bias/m/Read/ReadVariableOp-Adam/De2DTrans_3/kernel/m/Read/ReadVariableOp+Adam/De2DTrans_3/bias/m/Read/ReadVariableOp3Adam/Conv2DTrans_recon/kernel/m/Read/ReadVariableOp1Adam/Conv2DTrans_recon/bias/m/Read/ReadVariableOp*Adam/Conv2D_1/kernel/v/Read/ReadVariableOp(Adam/Conv2D_1/bias/v/Read/ReadVariableOp*Adam/Conv2D_2/kernel/v/Read/ReadVariableOp(Adam/Conv2D_2/bias/v/Read/ReadVariableOp*Adam/Conv2D_3/kernel/v/Read/ReadVariableOp(Adam/Conv2D_3/bias/v/Read/ReadVariableOp,Adam/DeConv2D_1/kernel/v/Read/ReadVariableOp*Adam/DeConv2D_1/bias/v/Read/ReadVariableOp-Adam/De2DTrans_2/kernel/v/Read/ReadVariableOp+Adam/De2DTrans_2/bias/v/Read/ReadVariableOp-Adam/De2DTrans_3/kernel/v/Read/ReadVariableOp+Adam/De2DTrans_3/bias/v/Read/ReadVariableOp3Adam/Conv2DTrans_recon/kernel/v/Read/ReadVariableOp1Adam/Conv2DTrans_recon/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_34543
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv2D_1/kernelConv2D_1/biasConv2D_2/kernelConv2D_2/biasConv2D_3/kernelConv2D_3/biasDeConv2D_1/kernelDeConv2D_1/biasDe2DTrans_2/kernelDe2DTrans_2/biasDe2DTrans_3/kernelDe2DTrans_3/biasConv2DTrans_recon/kernelConv2DTrans_recon/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv2D_1/kernel/mAdam/Conv2D_1/bias/mAdam/Conv2D_2/kernel/mAdam/Conv2D_2/bias/mAdam/Conv2D_3/kernel/mAdam/Conv2D_3/bias/mAdam/DeConv2D_1/kernel/mAdam/DeConv2D_1/bias/mAdam/De2DTrans_2/kernel/mAdam/De2DTrans_2/bias/mAdam/De2DTrans_3/kernel/mAdam/De2DTrans_3/bias/mAdam/Conv2DTrans_recon/kernel/mAdam/Conv2DTrans_recon/bias/mAdam/Conv2D_1/kernel/vAdam/Conv2D_1/bias/vAdam/Conv2D_2/kernel/vAdam/Conv2D_2/bias/vAdam/Conv2D_3/kernel/vAdam/Conv2D_3/bias/vAdam/DeConv2D_1/kernel/vAdam/DeConv2D_1/bias/vAdam/De2DTrans_2/kernel/vAdam/De2DTrans_2/bias/vAdam/De2DTrans_3/kernel/vAdam/De2DTrans_3/bias/vAdam/Conv2DTrans_recon/kernel/vAdam/Conv2DTrans_recon/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_34706??
?
E
)__inference_MaxPool_3_layer_call_fn_33379

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_3_layer_call_and_return_conditional_losses_333732
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_UpSample_3_layer_call_fn_33526

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_3_layer_call_and_return_conditional_losses_335202
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?$
?
E__inference_DeConv2D_1_layer_call_and_return_conditional_losses_33433

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_MaxPool_2_layer_call_fn_33367

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_2_layer_call_and_return_conditional_losses_333612
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_MaxPool_3_layer_call_and_return_conditional_losses_33373

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_UpSample_1_layer_call_and_return_conditional_losses_33392

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?j
?
__inference__traced_save_34543
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop0
,savev2_deconv2d_1_kernel_read_readvariableop.
*savev2_deconv2d_1_bias_read_readvariableop1
-savev2_de2dtrans_2_kernel_read_readvariableop/
+savev2_de2dtrans_2_bias_read_readvariableop1
-savev2_de2dtrans_3_kernel_read_readvariableop/
+savev2_de2dtrans_3_bias_read_readvariableop7
3savev2_conv2dtrans_recon_kernel_read_readvariableop5
1savev2_conv2dtrans_recon_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop7
3savev2_adam_deconv2d_1_kernel_m_read_readvariableop5
1savev2_adam_deconv2d_1_bias_m_read_readvariableop8
4savev2_adam_de2dtrans_2_kernel_m_read_readvariableop6
2savev2_adam_de2dtrans_2_bias_m_read_readvariableop8
4savev2_adam_de2dtrans_3_kernel_m_read_readvariableop6
2savev2_adam_de2dtrans_3_bias_m_read_readvariableop>
:savev2_adam_conv2dtrans_recon_kernel_m_read_readvariableop<
8savev2_adam_conv2dtrans_recon_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop7
3savev2_adam_deconv2d_1_kernel_v_read_readvariableop5
1savev2_adam_deconv2d_1_bias_v_read_readvariableop8
4savev2_adam_de2dtrans_2_kernel_v_read_readvariableop6
2savev2_adam_de2dtrans_2_bias_v_read_readvariableop8
4savev2_adam_de2dtrans_3_kernel_v_read_readvariableop6
2savev2_adam_de2dtrans_3_bias_v_read_readvariableop>
:savev2_adam_conv2dtrans_recon_kernel_v_read_readvariableop<
8savev2_adam_conv2dtrans_recon_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop,savev2_deconv2d_1_kernel_read_readvariableop*savev2_deconv2d_1_bias_read_readvariableop-savev2_de2dtrans_2_kernel_read_readvariableop+savev2_de2dtrans_2_bias_read_readvariableop-savev2_de2dtrans_3_kernel_read_readvariableop+savev2_de2dtrans_3_bias_read_readvariableop3savev2_conv2dtrans_recon_kernel_read_readvariableop1savev2_conv2dtrans_recon_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop3savev2_adam_deconv2d_1_kernel_m_read_readvariableop1savev2_adam_deconv2d_1_bias_m_read_readvariableop4savev2_adam_de2dtrans_2_kernel_m_read_readvariableop2savev2_adam_de2dtrans_2_bias_m_read_readvariableop4savev2_adam_de2dtrans_3_kernel_m_read_readvariableop2savev2_adam_de2dtrans_3_bias_m_read_readvariableop:savev2_adam_conv2dtrans_recon_kernel_m_read_readvariableop8savev2_adam_conv2dtrans_recon_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop3savev2_adam_deconv2d_1_kernel_v_read_readvariableop1savev2_adam_deconv2d_1_bias_v_read_readvariableop4savev2_adam_de2dtrans_2_kernel_v_read_readvariableop2savev2_adam_de2dtrans_2_bias_v_read_readvariableop4savev2_adam_de2dtrans_3_kernel_v_read_readvariableop2savev2_adam_de2dtrans_3_bias_v_read_readvariableop:savev2_adam_conv2dtrans_recon_kernel_v_read_readvariableop8savev2_adam_conv2dtrans_recon_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@?:?:??:?:@?:@: @: : :: : : : : : : : : : : : @:@:@?:?:??:?:@?:@: @: : :: : : @:@:@?:?:??:?:@?:@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:-	)
'
_output_shapes
:@?: 


_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:- )
'
_output_shapes
:@?: !

_output_shapes
:@:,"(
&
_output_shapes
: @: #

_output_shapes
: :,$(
&
_output_shapes
: : %

_output_shapes
::,&(
&
_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
: @: )

_output_shapes
:@:-*)
'
_output_shapes
:@?:!+

_output_shapes	
:?:.,*
(
_output_shapes
:??:!-

_output_shapes	
:?:-.)
'
_output_shapes
:@?: /

_output_shapes
:@:,0(
&
_output_shapes
: @: 1

_output_shapes
: :,2(
&
_output_shapes
: : 3

_output_shapes
::4

_output_shapes
: 
??
?
!__inference__traced_restore_34706
file_prefix$
 assignvariableop_conv2d_1_kernel$
 assignvariableop_1_conv2d_1_bias&
"assignvariableop_2_conv2d_2_kernel$
 assignvariableop_3_conv2d_2_bias&
"assignvariableop_4_conv2d_3_kernel$
 assignvariableop_5_conv2d_3_bias(
$assignvariableop_6_deconv2d_1_kernel&
"assignvariableop_7_deconv2d_1_bias)
%assignvariableop_8_de2dtrans_2_kernel'
#assignvariableop_9_de2dtrans_2_bias*
&assignvariableop_10_de2dtrans_3_kernel(
$assignvariableop_11_de2dtrans_3_bias0
,assignvariableop_12_conv2dtrans_recon_kernel.
*assignvariableop_13_conv2dtrans_recon_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate
assignvariableop_19_total
assignvariableop_20_count
assignvariableop_21_total_1
assignvariableop_22_count_1.
*assignvariableop_23_adam_conv2d_1_kernel_m,
(assignvariableop_24_adam_conv2d_1_bias_m.
*assignvariableop_25_adam_conv2d_2_kernel_m,
(assignvariableop_26_adam_conv2d_2_bias_m.
*assignvariableop_27_adam_conv2d_3_kernel_m,
(assignvariableop_28_adam_conv2d_3_bias_m0
,assignvariableop_29_adam_deconv2d_1_kernel_m.
*assignvariableop_30_adam_deconv2d_1_bias_m1
-assignvariableop_31_adam_de2dtrans_2_kernel_m/
+assignvariableop_32_adam_de2dtrans_2_bias_m1
-assignvariableop_33_adam_de2dtrans_3_kernel_m/
+assignvariableop_34_adam_de2dtrans_3_bias_m7
3assignvariableop_35_adam_conv2dtrans_recon_kernel_m5
1assignvariableop_36_adam_conv2dtrans_recon_bias_m.
*assignvariableop_37_adam_conv2d_1_kernel_v,
(assignvariableop_38_adam_conv2d_1_bias_v.
*assignvariableop_39_adam_conv2d_2_kernel_v,
(assignvariableop_40_adam_conv2d_2_bias_v.
*assignvariableop_41_adam_conv2d_3_kernel_v,
(assignvariableop_42_adam_conv2d_3_bias_v0
,assignvariableop_43_adam_deconv2d_1_kernel_v.
*assignvariableop_44_adam_deconv2d_1_bias_v1
-assignvariableop_45_adam_de2dtrans_2_kernel_v/
+assignvariableop_46_adam_de2dtrans_2_bias_v1
-assignvariableop_47_adam_de2dtrans_3_kernel_v/
+assignvariableop_48_adam_de2dtrans_3_bias_v7
3assignvariableop_49_adam_conv2dtrans_recon_kernel_v5
1assignvariableop_50_adam_conv2dtrans_recon_bias_v
identity_52??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_deconv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_deconv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_de2dtrans_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_de2dtrans_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_de2dtrans_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_de2dtrans_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp,assignvariableop_12_conv2dtrans_recon_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp*assignvariableop_13_conv2dtrans_recon_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_deconv2d_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_deconv2d_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_de2dtrans_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_de2dtrans_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp-assignvariableop_33_adam_de2dtrans_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_de2dtrans_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp3assignvariableop_35_adam_conv2dtrans_recon_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp1assignvariableop_36_adam_conv2dtrans_recon_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_deconv2d_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_deconv2d_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp-assignvariableop_45_adam_de2dtrans_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_de2dtrans_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_de2dtrans_3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_de2dtrans_3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adam_conv2dtrans_recon_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp1assignvariableop_50_adam_conv2dtrans_recon_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51?	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_52"#
identity_52Identity_52:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?:
?
E__inference_sequential_layer_call_and_return_conditional_losses_33732
conv2d_1_input
conv2d_1_33646
conv2d_1_33648
conv2d_2_33674
conv2d_2_33676
conv2d_3_33702
conv2d_3_33704
deconv2d_1_33709
deconv2d_1_33711
de2dtrans_2_33715
de2dtrans_2_33717
de2dtrans_3_33721
de2dtrans_3_33723
conv2dtrans_recon_33726
conv2dtrans_recon_33728
identity??)Conv2DTrans_recon/StatefulPartitionedCall? Conv2D_1/StatefulPartitionedCall? Conv2D_2/StatefulPartitionedCall? Conv2D_3/StatefulPartitionedCall?#De2DTrans_2/StatefulPartitionedCall?#De2DTrans_3/StatefulPartitionedCall?"DeConv2D_1/StatefulPartitionedCall?
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_33646conv2d_1_33648*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_336352"
 Conv2D_1/StatefulPartitionedCall?
MaxPool_1/PartitionedCallPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_1_layer_call_and_return_conditional_losses_333492
MaxPool_1/PartitionedCall?
 Conv2D_2/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_1/PartitionedCall:output:0conv2d_2_33674conv2d_2_33676*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_336632"
 Conv2D_2/StatefulPartitionedCall?
MaxPool_2/PartitionedCallPartitionedCall)Conv2D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_2_layer_call_and_return_conditional_losses_333612
MaxPool_2/PartitionedCall?
 Conv2D_3/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_2/PartitionedCall:output:0conv2d_3_33702conv2d_3_33704*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_336912"
 Conv2D_3/StatefulPartitionedCall?
MaxPool_3/PartitionedCallPartitionedCall)Conv2D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_3_layer_call_and_return_conditional_losses_333732
MaxPool_3/PartitionedCall?
UpSample_1/PartitionedCallPartitionedCall"MaxPool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_1_layer_call_and_return_conditional_losses_333922
UpSample_1/PartitionedCall?
"DeConv2D_1/StatefulPartitionedCallStatefulPartitionedCall#UpSample_1/PartitionedCall:output:0deconv2d_1_33709deconv2d_1_33711*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_DeConv2D_1_layer_call_and_return_conditional_losses_334332$
"DeConv2D_1/StatefulPartitionedCall?
UpSample_2/PartitionedCallPartitionedCall+DeConv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_2_layer_call_and_return_conditional_losses_334562
UpSample_2/PartitionedCall?
#De2DTrans_2/StatefulPartitionedCallStatefulPartitionedCall#UpSample_2/PartitionedCall:output:0de2dtrans_2_33715de2dtrans_2_33717*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_2_layer_call_and_return_conditional_losses_334972%
#De2DTrans_2/StatefulPartitionedCall?
UpSample_3/PartitionedCallPartitionedCall,De2DTrans_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_3_layer_call_and_return_conditional_losses_335202
UpSample_3/PartitionedCall?
#De2DTrans_3/StatefulPartitionedCallStatefulPartitionedCall#UpSample_3/PartitionedCall:output:0de2dtrans_3_33721de2dtrans_3_33723*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_3_layer_call_and_return_conditional_losses_335612%
#De2DTrans_3/StatefulPartitionedCall?
)Conv2DTrans_recon/StatefulPartitionedCallStatefulPartitionedCall,De2DTrans_3/StatefulPartitionedCall:output:0conv2dtrans_recon_33726conv2dtrans_recon_33728*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_Conv2DTrans_recon_layer_call_and_return_conditional_losses_336102+
)Conv2DTrans_recon/StatefulPartitionedCall?
IdentityIdentity2Conv2DTrans_recon/StatefulPartitionedCall:output:0*^Conv2DTrans_recon/StatefulPartitionedCall!^Conv2D_1/StatefulPartitionedCall!^Conv2D_2/StatefulPartitionedCall!^Conv2D_3/StatefulPartitionedCall$^De2DTrans_2/StatefulPartitionedCall$^De2DTrans_3/StatefulPartitionedCall#^DeConv2D_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2V
)Conv2DTrans_recon/StatefulPartitionedCall)Conv2DTrans_recon/StatefulPartitionedCall2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2D
 Conv2D_2/StatefulPartitionedCall Conv2D_2/StatefulPartitionedCall2D
 Conv2D_3/StatefulPartitionedCall Conv2D_3/StatefulPartitionedCall2J
#De2DTrans_2/StatefulPartitionedCall#De2DTrans_2/StatefulPartitionedCall2J
#De2DTrans_3/StatefulPartitionedCall#De2DTrans_3/StatefulPartitionedCall2H
"DeConv2D_1/StatefulPartitionedCall"DeConv2D_1/StatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameConv2D_1_input
?

?
*__inference_sequential_layer_call_fn_34274

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_338252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_UpSample_3_layer_call_and_return_conditional_losses_33520

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_33343
conv2d_1_input6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource6
2sequential_conv2d_3_conv2d_readvariableop_resource7
3sequential_conv2d_3_biasadd_readvariableop_resourceB
>sequential_deconv2d_1_conv2d_transpose_readvariableop_resource9
5sequential_deconv2d_1_biasadd_readvariableop_resourceC
?sequential_de2dtrans_2_conv2d_transpose_readvariableop_resource:
6sequential_de2dtrans_2_biasadd_readvariableop_resourceC
?sequential_de2dtrans_3_conv2d_transpose_readvariableop_resource:
6sequential_de2dtrans_3_biasadd_readvariableop_resourceI
Esequential_conv2dtrans_recon_conv2d_transpose_readvariableop_resource@
<sequential_conv2dtrans_recon_biasadd_readvariableop_resource
identity??3sequential/Conv2DTrans_recon/BiasAdd/ReadVariableOp?<sequential/Conv2DTrans_recon/conv2d_transpose/ReadVariableOp?*sequential/Conv2D_1/BiasAdd/ReadVariableOp?)sequential/Conv2D_1/Conv2D/ReadVariableOp?*sequential/Conv2D_2/BiasAdd/ReadVariableOp?)sequential/Conv2D_2/Conv2D/ReadVariableOp?*sequential/Conv2D_3/BiasAdd/ReadVariableOp?)sequential/Conv2D_3/Conv2D/ReadVariableOp?-sequential/De2DTrans_2/BiasAdd/ReadVariableOp?6sequential/De2DTrans_2/conv2d_transpose/ReadVariableOp?-sequential/De2DTrans_3/BiasAdd/ReadVariableOp?6sequential/De2DTrans_3/conv2d_transpose/ReadVariableOp?,sequential/DeConv2D_1/BiasAdd/ReadVariableOp?5sequential/DeConv2D_1/conv2d_transpose/ReadVariableOp?
)sequential/Conv2D_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)sequential/Conv2D_1/Conv2D/ReadVariableOp?
sequential/Conv2D_1/Conv2DConv2Dconv2d_1_input1sequential/Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
sequential/Conv2D_1/Conv2D?
*sequential/Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/Conv2D_1/BiasAdd/ReadVariableOp?
sequential/Conv2D_1/BiasAddBiasAdd#sequential/Conv2D_1/Conv2D:output:02sequential/Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
sequential/Conv2D_1/BiasAdd?
sequential/Conv2D_1/ReluRelu$sequential/Conv2D_1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
sequential/Conv2D_1/Relu?
sequential/MaxPool_1/MaxPoolMaxPool&sequential/Conv2D_1/Relu:activations:0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2
sequential/MaxPool_1/MaxPool?
)sequential/Conv2D_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/Conv2D_2/Conv2D/ReadVariableOp?
sequential/Conv2D_2/Conv2DConv2D%sequential/MaxPool_1/MaxPool:output:01sequential/Conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
sequential/Conv2D_2/Conv2D?
*sequential/Conv2D_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/Conv2D_2/BiasAdd/ReadVariableOp?
sequential/Conv2D_2/BiasAddBiasAdd#sequential/Conv2D_2/Conv2D:output:02sequential/Conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
sequential/Conv2D_2/BiasAdd?
sequential/Conv2D_2/ReluRelu$sequential/Conv2D_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
sequential/Conv2D_2/Relu?
sequential/MaxPool_2/MaxPoolMaxPool&sequential/Conv2D_2/Relu:activations:0*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
2
sequential/MaxPool_2/MaxPool?
)sequential/Conv2D_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02+
)sequential/Conv2D_3/Conv2D/ReadVariableOp?
sequential/Conv2D_3/Conv2DConv2D%sequential/MaxPool_2/MaxPool:output:01sequential/Conv2D_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
sequential/Conv2D_3/Conv2D?
*sequential/Conv2D_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*sequential/Conv2D_3/BiasAdd/ReadVariableOp?
sequential/Conv2D_3/BiasAddBiasAdd#sequential/Conv2D_3/Conv2D:output:02sequential/Conv2D_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
sequential/Conv2D_3/BiasAdd?
sequential/Conv2D_3/ReluRelu$sequential/Conv2D_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
sequential/Conv2D_3/Relu?
sequential/MaxPool_3/MaxPoolMaxPool&sequential/Conv2D_3/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2
sequential/MaxPool_3/MaxPool?
sequential/UpSample_1/ShapeShape%sequential/MaxPool_3/MaxPool:output:0*
T0*
_output_shapes
:2
sequential/UpSample_1/Shape?
)sequential/UpSample_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)sequential/UpSample_1/strided_slice/stack?
+sequential/UpSample_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/UpSample_1/strided_slice/stack_1?
+sequential/UpSample_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/UpSample_1/strided_slice/stack_2?
#sequential/UpSample_1/strided_sliceStridedSlice$sequential/UpSample_1/Shape:output:02sequential/UpSample_1/strided_slice/stack:output:04sequential/UpSample_1/strided_slice/stack_1:output:04sequential/UpSample_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#sequential/UpSample_1/strided_slice?
sequential/UpSample_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
sequential/UpSample_1/Const?
sequential/UpSample_1/mulMul,sequential/UpSample_1/strided_slice:output:0$sequential/UpSample_1/Const:output:0*
T0*
_output_shapes
:2
sequential/UpSample_1/mul?
+sequential/UpSample_1/resize/ResizeBilinearResizeBilinear%sequential/MaxPool_3/MaxPool:output:0sequential/UpSample_1/mul:z:0*
T0*0
_output_shapes
:?????????@@?*
half_pixel_centers(2-
+sequential/UpSample_1/resize/ResizeBilinear?
sequential/DeConv2D_1/ShapeShape<sequential/UpSample_1/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
sequential/DeConv2D_1/Shape?
)sequential/DeConv2D_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/DeConv2D_1/strided_slice/stack?
+sequential/DeConv2D_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/DeConv2D_1/strided_slice/stack_1?
+sequential/DeConv2D_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/DeConv2D_1/strided_slice/stack_2?
#sequential/DeConv2D_1/strided_sliceStridedSlice$sequential/DeConv2D_1/Shape:output:02sequential/DeConv2D_1/strided_slice/stack:output:04sequential/DeConv2D_1/strided_slice/stack_1:output:04sequential/DeConv2D_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/DeConv2D_1/strided_slice?
sequential/DeConv2D_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
sequential/DeConv2D_1/stack/1?
sequential/DeConv2D_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
sequential/DeConv2D_1/stack/2?
sequential/DeConv2D_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
sequential/DeConv2D_1/stack/3?
sequential/DeConv2D_1/stackPack,sequential/DeConv2D_1/strided_slice:output:0&sequential/DeConv2D_1/stack/1:output:0&sequential/DeConv2D_1/stack/2:output:0&sequential/DeConv2D_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
sequential/DeConv2D_1/stack?
+sequential/DeConv2D_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/DeConv2D_1/strided_slice_1/stack?
-sequential/DeConv2D_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/DeConv2D_1/strided_slice_1/stack_1?
-sequential/DeConv2D_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/DeConv2D_1/strided_slice_1/stack_2?
%sequential/DeConv2D_1/strided_slice_1StridedSlice$sequential/DeConv2D_1/stack:output:04sequential/DeConv2D_1/strided_slice_1/stack:output:06sequential/DeConv2D_1/strided_slice_1/stack_1:output:06sequential/DeConv2D_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/DeConv2D_1/strided_slice_1?
5sequential/DeConv2D_1/conv2d_transpose/ReadVariableOpReadVariableOp>sequential_deconv2d_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype027
5sequential/DeConv2D_1/conv2d_transpose/ReadVariableOp?
&sequential/DeConv2D_1/conv2d_transposeConv2DBackpropInput$sequential/DeConv2D_1/stack:output:0=sequential/DeConv2D_1/conv2d_transpose/ReadVariableOp:value:0<sequential/UpSample_1/resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2(
&sequential/DeConv2D_1/conv2d_transpose?
,sequential/DeConv2D_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_deconv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential/DeConv2D_1/BiasAdd/ReadVariableOp?
sequential/DeConv2D_1/BiasAddBiasAdd/sequential/DeConv2D_1/conv2d_transpose:output:04sequential/DeConv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
sequential/DeConv2D_1/BiasAdd?
sequential/DeConv2D_1/ReluRelu&sequential/DeConv2D_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
sequential/DeConv2D_1/Relu?
sequential/UpSample_2/ShapeShape(sequential/DeConv2D_1/Relu:activations:0*
T0*
_output_shapes
:2
sequential/UpSample_2/Shape?
)sequential/UpSample_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)sequential/UpSample_2/strided_slice/stack?
+sequential/UpSample_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/UpSample_2/strided_slice/stack_1?
+sequential/UpSample_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/UpSample_2/strided_slice/stack_2?
#sequential/UpSample_2/strided_sliceStridedSlice$sequential/UpSample_2/Shape:output:02sequential/UpSample_2/strided_slice/stack:output:04sequential/UpSample_2/strided_slice/stack_1:output:04sequential/UpSample_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#sequential/UpSample_2/strided_slice?
sequential/UpSample_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
sequential/UpSample_2/Const?
sequential/UpSample_2/mulMul,sequential/UpSample_2/strided_slice:output:0$sequential/UpSample_2/Const:output:0*
T0*
_output_shapes
:2
sequential/UpSample_2/mul?
+sequential/UpSample_2/resize/ResizeBilinearResizeBilinear(sequential/DeConv2D_1/Relu:activations:0sequential/UpSample_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2-
+sequential/UpSample_2/resize/ResizeBilinear?
sequential/De2DTrans_2/ShapeShape<sequential/UpSample_2/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
sequential/De2DTrans_2/Shape?
*sequential/De2DTrans_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential/De2DTrans_2/strided_slice/stack?
,sequential/De2DTrans_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/De2DTrans_2/strided_slice/stack_1?
,sequential/De2DTrans_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/De2DTrans_2/strided_slice/stack_2?
$sequential/De2DTrans_2/strided_sliceStridedSlice%sequential/De2DTrans_2/Shape:output:03sequential/De2DTrans_2/strided_slice/stack:output:05sequential/De2DTrans_2/strided_slice/stack_1:output:05sequential/De2DTrans_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential/De2DTrans_2/strided_slice?
sequential/De2DTrans_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2 
sequential/De2DTrans_2/stack/1?
sequential/De2DTrans_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2 
sequential/De2DTrans_2/stack/2?
sequential/De2DTrans_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2 
sequential/De2DTrans_2/stack/3?
sequential/De2DTrans_2/stackPack-sequential/De2DTrans_2/strided_slice:output:0'sequential/De2DTrans_2/stack/1:output:0'sequential/De2DTrans_2/stack/2:output:0'sequential/De2DTrans_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
sequential/De2DTrans_2/stack?
,sequential/De2DTrans_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential/De2DTrans_2/strided_slice_1/stack?
.sequential/De2DTrans_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/De2DTrans_2/strided_slice_1/stack_1?
.sequential/De2DTrans_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/De2DTrans_2/strided_slice_1/stack_2?
&sequential/De2DTrans_2/strided_slice_1StridedSlice%sequential/De2DTrans_2/stack:output:05sequential/De2DTrans_2/strided_slice_1/stack:output:07sequential/De2DTrans_2/strided_slice_1/stack_1:output:07sequential/De2DTrans_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential/De2DTrans_2/strided_slice_1?
6sequential/De2DTrans_2/conv2d_transpose/ReadVariableOpReadVariableOp?sequential_de2dtrans_2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype028
6sequential/De2DTrans_2/conv2d_transpose/ReadVariableOp?
'sequential/De2DTrans_2/conv2d_transposeConv2DBackpropInput%sequential/De2DTrans_2/stack:output:0>sequential/De2DTrans_2/conv2d_transpose/ReadVariableOp:value:0<sequential/UpSample_2/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2)
'sequential/De2DTrans_2/conv2d_transpose?
-sequential/De2DTrans_2/BiasAdd/ReadVariableOpReadVariableOp6sequential_de2dtrans_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential/De2DTrans_2/BiasAdd/ReadVariableOp?
sequential/De2DTrans_2/BiasAddBiasAdd0sequential/De2DTrans_2/conv2d_transpose:output:05sequential/De2DTrans_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2 
sequential/De2DTrans_2/BiasAdd?
sequential/De2DTrans_2/ReluRelu'sequential/De2DTrans_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
sequential/De2DTrans_2/Relu?
sequential/UpSample_3/ShapeShape)sequential/De2DTrans_2/Relu:activations:0*
T0*
_output_shapes
:2
sequential/UpSample_3/Shape?
)sequential/UpSample_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)sequential/UpSample_3/strided_slice/stack?
+sequential/UpSample_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/UpSample_3/strided_slice/stack_1?
+sequential/UpSample_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/UpSample_3/strided_slice/stack_2?
#sequential/UpSample_3/strided_sliceStridedSlice$sequential/UpSample_3/Shape:output:02sequential/UpSample_3/strided_slice/stack:output:04sequential/UpSample_3/strided_slice/stack_1:output:04sequential/UpSample_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#sequential/UpSample_3/strided_slice?
sequential/UpSample_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
sequential/UpSample_3/Const?
sequential/UpSample_3/mulMul,sequential/UpSample_3/strided_slice:output:0$sequential/UpSample_3/Const:output:0*
T0*
_output_shapes
:2
sequential/UpSample_3/mul?
+sequential/UpSample_3/resize/ResizeBilinearResizeBilinear)sequential/De2DTrans_2/Relu:activations:0sequential/UpSample_3/mul:z:0*
T0*1
_output_shapes
:???????????@*
half_pixel_centers(2-
+sequential/UpSample_3/resize/ResizeBilinear?
sequential/De2DTrans_3/ShapeShape<sequential/UpSample_3/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
sequential/De2DTrans_3/Shape?
*sequential/De2DTrans_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential/De2DTrans_3/strided_slice/stack?
,sequential/De2DTrans_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/De2DTrans_3/strided_slice/stack_1?
,sequential/De2DTrans_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/De2DTrans_3/strided_slice/stack_2?
$sequential/De2DTrans_3/strided_sliceStridedSlice%sequential/De2DTrans_3/Shape:output:03sequential/De2DTrans_3/strided_slice/stack:output:05sequential/De2DTrans_3/strided_slice/stack_1:output:05sequential/De2DTrans_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential/De2DTrans_3/strided_slice?
sequential/De2DTrans_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2 
sequential/De2DTrans_3/stack/1?
sequential/De2DTrans_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2 
sequential/De2DTrans_3/stack/2?
sequential/De2DTrans_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2 
sequential/De2DTrans_3/stack/3?
sequential/De2DTrans_3/stackPack-sequential/De2DTrans_3/strided_slice:output:0'sequential/De2DTrans_3/stack/1:output:0'sequential/De2DTrans_3/stack/2:output:0'sequential/De2DTrans_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
sequential/De2DTrans_3/stack?
,sequential/De2DTrans_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential/De2DTrans_3/strided_slice_1/stack?
.sequential/De2DTrans_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/De2DTrans_3/strided_slice_1/stack_1?
.sequential/De2DTrans_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/De2DTrans_3/strided_slice_1/stack_2?
&sequential/De2DTrans_3/strided_slice_1StridedSlice%sequential/De2DTrans_3/stack:output:05sequential/De2DTrans_3/strided_slice_1/stack:output:07sequential/De2DTrans_3/strided_slice_1/stack_1:output:07sequential/De2DTrans_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential/De2DTrans_3/strided_slice_1?
6sequential/De2DTrans_3/conv2d_transpose/ReadVariableOpReadVariableOp?sequential_de2dtrans_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype028
6sequential/De2DTrans_3/conv2d_transpose/ReadVariableOp?
'sequential/De2DTrans_3/conv2d_transposeConv2DBackpropInput%sequential/De2DTrans_3/stack:output:0>sequential/De2DTrans_3/conv2d_transpose/ReadVariableOp:value:0<sequential/UpSample_3/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2)
'sequential/De2DTrans_3/conv2d_transpose?
-sequential/De2DTrans_3/BiasAdd/ReadVariableOpReadVariableOp6sequential_de2dtrans_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential/De2DTrans_3/BiasAdd/ReadVariableOp?
sequential/De2DTrans_3/BiasAddBiasAdd0sequential/De2DTrans_3/conv2d_transpose:output:05sequential/De2DTrans_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2 
sequential/De2DTrans_3/BiasAdd?
sequential/De2DTrans_3/ReluRelu'sequential/De2DTrans_3/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
sequential/De2DTrans_3/Relu?
"sequential/Conv2DTrans_recon/ShapeShape)sequential/De2DTrans_3/Relu:activations:0*
T0*
_output_shapes
:2$
"sequential/Conv2DTrans_recon/Shape?
0sequential/Conv2DTrans_recon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/Conv2DTrans_recon/strided_slice/stack?
2sequential/Conv2DTrans_recon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/Conv2DTrans_recon/strided_slice/stack_1?
2sequential/Conv2DTrans_recon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/Conv2DTrans_recon/strided_slice/stack_2?
*sequential/Conv2DTrans_recon/strided_sliceStridedSlice+sequential/Conv2DTrans_recon/Shape:output:09sequential/Conv2DTrans_recon/strided_slice/stack:output:0;sequential/Conv2DTrans_recon/strided_slice/stack_1:output:0;sequential/Conv2DTrans_recon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential/Conv2DTrans_recon/strided_slice?
$sequential/Conv2DTrans_recon/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential/Conv2DTrans_recon/stack/1?
$sequential/Conv2DTrans_recon/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential/Conv2DTrans_recon/stack/2?
$sequential/Conv2DTrans_recon/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential/Conv2DTrans_recon/stack/3?
"sequential/Conv2DTrans_recon/stackPack3sequential/Conv2DTrans_recon/strided_slice:output:0-sequential/Conv2DTrans_recon/stack/1:output:0-sequential/Conv2DTrans_recon/stack/2:output:0-sequential/Conv2DTrans_recon/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"sequential/Conv2DTrans_recon/stack?
2sequential/Conv2DTrans_recon/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential/Conv2DTrans_recon/strided_slice_1/stack?
4sequential/Conv2DTrans_recon/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential/Conv2DTrans_recon/strided_slice_1/stack_1?
4sequential/Conv2DTrans_recon/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential/Conv2DTrans_recon/strided_slice_1/stack_2?
,sequential/Conv2DTrans_recon/strided_slice_1StridedSlice+sequential/Conv2DTrans_recon/stack:output:0;sequential/Conv2DTrans_recon/strided_slice_1/stack:output:0=sequential/Conv2DTrans_recon/strided_slice_1/stack_1:output:0=sequential/Conv2DTrans_recon/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential/Conv2DTrans_recon/strided_slice_1?
<sequential/Conv2DTrans_recon/conv2d_transpose/ReadVariableOpReadVariableOpEsequential_conv2dtrans_recon_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02>
<sequential/Conv2DTrans_recon/conv2d_transpose/ReadVariableOp?
-sequential/Conv2DTrans_recon/conv2d_transposeConv2DBackpropInput+sequential/Conv2DTrans_recon/stack:output:0Dsequential/Conv2DTrans_recon/conv2d_transpose/ReadVariableOp:value:0)sequential/De2DTrans_3/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2/
-sequential/Conv2DTrans_recon/conv2d_transpose?
3sequential/Conv2DTrans_recon/BiasAdd/ReadVariableOpReadVariableOp<sequential_conv2dtrans_recon_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential/Conv2DTrans_recon/BiasAdd/ReadVariableOp?
$sequential/Conv2DTrans_recon/BiasAddBiasAdd6sequential/Conv2DTrans_recon/conv2d_transpose:output:0;sequential/Conv2DTrans_recon/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2&
$sequential/Conv2DTrans_recon/BiasAdd?
!sequential/Conv2DTrans_recon/ReluRelu-sequential/Conv2DTrans_recon/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2#
!sequential/Conv2DTrans_recon/Relu?
IdentityIdentity/sequential/Conv2DTrans_recon/Relu:activations:04^sequential/Conv2DTrans_recon/BiasAdd/ReadVariableOp=^sequential/Conv2DTrans_recon/conv2d_transpose/ReadVariableOp+^sequential/Conv2D_1/BiasAdd/ReadVariableOp*^sequential/Conv2D_1/Conv2D/ReadVariableOp+^sequential/Conv2D_2/BiasAdd/ReadVariableOp*^sequential/Conv2D_2/Conv2D/ReadVariableOp+^sequential/Conv2D_3/BiasAdd/ReadVariableOp*^sequential/Conv2D_3/Conv2D/ReadVariableOp.^sequential/De2DTrans_2/BiasAdd/ReadVariableOp7^sequential/De2DTrans_2/conv2d_transpose/ReadVariableOp.^sequential/De2DTrans_3/BiasAdd/ReadVariableOp7^sequential/De2DTrans_3/conv2d_transpose/ReadVariableOp-^sequential/DeConv2D_1/BiasAdd/ReadVariableOp6^sequential/DeConv2D_1/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2j
3sequential/Conv2DTrans_recon/BiasAdd/ReadVariableOp3sequential/Conv2DTrans_recon/BiasAdd/ReadVariableOp2|
<sequential/Conv2DTrans_recon/conv2d_transpose/ReadVariableOp<sequential/Conv2DTrans_recon/conv2d_transpose/ReadVariableOp2X
*sequential/Conv2D_1/BiasAdd/ReadVariableOp*sequential/Conv2D_1/BiasAdd/ReadVariableOp2V
)sequential/Conv2D_1/Conv2D/ReadVariableOp)sequential/Conv2D_1/Conv2D/ReadVariableOp2X
*sequential/Conv2D_2/BiasAdd/ReadVariableOp*sequential/Conv2D_2/BiasAdd/ReadVariableOp2V
)sequential/Conv2D_2/Conv2D/ReadVariableOp)sequential/Conv2D_2/Conv2D/ReadVariableOp2X
*sequential/Conv2D_3/BiasAdd/ReadVariableOp*sequential/Conv2D_3/BiasAdd/ReadVariableOp2V
)sequential/Conv2D_3/Conv2D/ReadVariableOp)sequential/Conv2D_3/Conv2D/ReadVariableOp2^
-sequential/De2DTrans_2/BiasAdd/ReadVariableOp-sequential/De2DTrans_2/BiasAdd/ReadVariableOp2p
6sequential/De2DTrans_2/conv2d_transpose/ReadVariableOp6sequential/De2DTrans_2/conv2d_transpose/ReadVariableOp2^
-sequential/De2DTrans_3/BiasAdd/ReadVariableOp-sequential/De2DTrans_3/BiasAdd/ReadVariableOp2p
6sequential/De2DTrans_3/conv2d_transpose/ReadVariableOp6sequential/De2DTrans_3/conv2d_transpose/ReadVariableOp2\
,sequential/DeConv2D_1/BiasAdd/ReadVariableOp,sequential/DeConv2D_1/BiasAdd/ReadVariableOp2n
5sequential/DeConv2D_1/conv2d_transpose/ReadVariableOp5sequential/DeConv2D_1/conv2d_transpose/ReadVariableOp:a ]
1
_output_shapes
:???????????
(
_user_specified_nameConv2D_1_input
?

?
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_33691

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?:
?
E__inference_sequential_layer_call_and_return_conditional_losses_33903

inputs
conv2d_1_33861
conv2d_1_33863
conv2d_2_33867
conv2d_2_33869
conv2d_3_33873
conv2d_3_33875
deconv2d_1_33880
deconv2d_1_33882
de2dtrans_2_33886
de2dtrans_2_33888
de2dtrans_3_33892
de2dtrans_3_33894
conv2dtrans_recon_33897
conv2dtrans_recon_33899
identity??)Conv2DTrans_recon/StatefulPartitionedCall? Conv2D_1/StatefulPartitionedCall? Conv2D_2/StatefulPartitionedCall? Conv2D_3/StatefulPartitionedCall?#De2DTrans_2/StatefulPartitionedCall?#De2DTrans_3/StatefulPartitionedCall?"DeConv2D_1/StatefulPartitionedCall?
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_33861conv2d_1_33863*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_336352"
 Conv2D_1/StatefulPartitionedCall?
MaxPool_1/PartitionedCallPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_1_layer_call_and_return_conditional_losses_333492
MaxPool_1/PartitionedCall?
 Conv2D_2/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_1/PartitionedCall:output:0conv2d_2_33867conv2d_2_33869*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_336632"
 Conv2D_2/StatefulPartitionedCall?
MaxPool_2/PartitionedCallPartitionedCall)Conv2D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_2_layer_call_and_return_conditional_losses_333612
MaxPool_2/PartitionedCall?
 Conv2D_3/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_2/PartitionedCall:output:0conv2d_3_33873conv2d_3_33875*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_336912"
 Conv2D_3/StatefulPartitionedCall?
MaxPool_3/PartitionedCallPartitionedCall)Conv2D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_3_layer_call_and_return_conditional_losses_333732
MaxPool_3/PartitionedCall?
UpSample_1/PartitionedCallPartitionedCall"MaxPool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_1_layer_call_and_return_conditional_losses_333922
UpSample_1/PartitionedCall?
"DeConv2D_1/StatefulPartitionedCallStatefulPartitionedCall#UpSample_1/PartitionedCall:output:0deconv2d_1_33880deconv2d_1_33882*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_DeConv2D_1_layer_call_and_return_conditional_losses_334332$
"DeConv2D_1/StatefulPartitionedCall?
UpSample_2/PartitionedCallPartitionedCall+DeConv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_2_layer_call_and_return_conditional_losses_334562
UpSample_2/PartitionedCall?
#De2DTrans_2/StatefulPartitionedCallStatefulPartitionedCall#UpSample_2/PartitionedCall:output:0de2dtrans_2_33886de2dtrans_2_33888*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_2_layer_call_and_return_conditional_losses_334972%
#De2DTrans_2/StatefulPartitionedCall?
UpSample_3/PartitionedCallPartitionedCall,De2DTrans_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_3_layer_call_and_return_conditional_losses_335202
UpSample_3/PartitionedCall?
#De2DTrans_3/StatefulPartitionedCallStatefulPartitionedCall#UpSample_3/PartitionedCall:output:0de2dtrans_3_33892de2dtrans_3_33894*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_3_layer_call_and_return_conditional_losses_335612%
#De2DTrans_3/StatefulPartitionedCall?
)Conv2DTrans_recon/StatefulPartitionedCallStatefulPartitionedCall,De2DTrans_3/StatefulPartitionedCall:output:0conv2dtrans_recon_33897conv2dtrans_recon_33899*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_Conv2DTrans_recon_layer_call_and_return_conditional_losses_336102+
)Conv2DTrans_recon/StatefulPartitionedCall?
IdentityIdentity2Conv2DTrans_recon/StatefulPartitionedCall:output:0*^Conv2DTrans_recon/StatefulPartitionedCall!^Conv2D_1/StatefulPartitionedCall!^Conv2D_2/StatefulPartitionedCall!^Conv2D_3/StatefulPartitionedCall$^De2DTrans_2/StatefulPartitionedCall$^De2DTrans_3/StatefulPartitionedCall#^DeConv2D_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2V
)Conv2DTrans_recon/StatefulPartitionedCall)Conv2DTrans_recon/StatefulPartitionedCall2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2D
 Conv2D_2/StatefulPartitionedCall Conv2D_2/StatefulPartitionedCall2D
 Conv2D_3/StatefulPartitionedCall Conv2D_3/StatefulPartitionedCall2J
#De2DTrans_2/StatefulPartitionedCall#De2DTrans_2/StatefulPartitionedCall2J
#De2DTrans_3/StatefulPartitionedCall#De2DTrans_3/StatefulPartitionedCall2H
"DeConv2D_1/StatefulPartitionedCall"DeConv2D_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?'
?
L__inference_Conv2DTrans_recon_layer_call_and_return_conditional_losses_33610

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
*__inference_sequential_layer_call_fn_33856
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_338252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameConv2D_1_input
?:
?
E__inference_sequential_layer_call_and_return_conditional_losses_33777
conv2d_1_input
conv2d_1_33735
conv2d_1_33737
conv2d_2_33741
conv2d_2_33743
conv2d_3_33747
conv2d_3_33749
deconv2d_1_33754
deconv2d_1_33756
de2dtrans_2_33760
de2dtrans_2_33762
de2dtrans_3_33766
de2dtrans_3_33768
conv2dtrans_recon_33771
conv2dtrans_recon_33773
identity??)Conv2DTrans_recon/StatefulPartitionedCall? Conv2D_1/StatefulPartitionedCall? Conv2D_2/StatefulPartitionedCall? Conv2D_3/StatefulPartitionedCall?#De2DTrans_2/StatefulPartitionedCall?#De2DTrans_3/StatefulPartitionedCall?"DeConv2D_1/StatefulPartitionedCall?
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_33735conv2d_1_33737*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_336352"
 Conv2D_1/StatefulPartitionedCall?
MaxPool_1/PartitionedCallPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_1_layer_call_and_return_conditional_losses_333492
MaxPool_1/PartitionedCall?
 Conv2D_2/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_1/PartitionedCall:output:0conv2d_2_33741conv2d_2_33743*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_336632"
 Conv2D_2/StatefulPartitionedCall?
MaxPool_2/PartitionedCallPartitionedCall)Conv2D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_2_layer_call_and_return_conditional_losses_333612
MaxPool_2/PartitionedCall?
 Conv2D_3/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_2/PartitionedCall:output:0conv2d_3_33747conv2d_3_33749*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_336912"
 Conv2D_3/StatefulPartitionedCall?
MaxPool_3/PartitionedCallPartitionedCall)Conv2D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_3_layer_call_and_return_conditional_losses_333732
MaxPool_3/PartitionedCall?
UpSample_1/PartitionedCallPartitionedCall"MaxPool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_1_layer_call_and_return_conditional_losses_333922
UpSample_1/PartitionedCall?
"DeConv2D_1/StatefulPartitionedCallStatefulPartitionedCall#UpSample_1/PartitionedCall:output:0deconv2d_1_33754deconv2d_1_33756*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_DeConv2D_1_layer_call_and_return_conditional_losses_334332$
"DeConv2D_1/StatefulPartitionedCall?
UpSample_2/PartitionedCallPartitionedCall+DeConv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_2_layer_call_and_return_conditional_losses_334562
UpSample_2/PartitionedCall?
#De2DTrans_2/StatefulPartitionedCallStatefulPartitionedCall#UpSample_2/PartitionedCall:output:0de2dtrans_2_33760de2dtrans_2_33762*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_2_layer_call_and_return_conditional_losses_334972%
#De2DTrans_2/StatefulPartitionedCall?
UpSample_3/PartitionedCallPartitionedCall,De2DTrans_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_3_layer_call_and_return_conditional_losses_335202
UpSample_3/PartitionedCall?
#De2DTrans_3/StatefulPartitionedCallStatefulPartitionedCall#UpSample_3/PartitionedCall:output:0de2dtrans_3_33766de2dtrans_3_33768*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_3_layer_call_and_return_conditional_losses_335612%
#De2DTrans_3/StatefulPartitionedCall?
)Conv2DTrans_recon/StatefulPartitionedCallStatefulPartitionedCall,De2DTrans_3/StatefulPartitionedCall:output:0conv2dtrans_recon_33771conv2dtrans_recon_33773*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_Conv2DTrans_recon_layer_call_and_return_conditional_losses_336102+
)Conv2DTrans_recon/StatefulPartitionedCall?
IdentityIdentity2Conv2DTrans_recon/StatefulPartitionedCall:output:0*^Conv2DTrans_recon/StatefulPartitionedCall!^Conv2D_1/StatefulPartitionedCall!^Conv2D_2/StatefulPartitionedCall!^Conv2D_3/StatefulPartitionedCall$^De2DTrans_2/StatefulPartitionedCall$^De2DTrans_3/StatefulPartitionedCall#^DeConv2D_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2V
)Conv2DTrans_recon/StatefulPartitionedCall)Conv2DTrans_recon/StatefulPartitionedCall2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2D
 Conv2D_2/StatefulPartitionedCall Conv2D_2/StatefulPartitionedCall2D
 Conv2D_3/StatefulPartitionedCall Conv2D_3/StatefulPartitionedCall2J
#De2DTrans_2/StatefulPartitionedCall#De2DTrans_2/StatefulPartitionedCall2J
#De2DTrans_3/StatefulPartitionedCall#De2DTrans_3/StatefulPartitionedCall2H
"DeConv2D_1/StatefulPartitionedCall"DeConv2D_1/StatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameConv2D_1_input
?

?
*__inference_sequential_layer_call_fn_34307

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_339032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_MaxPool_1_layer_call_and_return_conditional_losses_33349

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
*__inference_sequential_layer_call_fn_33934
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_339032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameConv2D_1_input
?$
?
F__inference_De2DTrans_2_layer_call_and_return_conditional_losses_33497

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_33977
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_333432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameConv2D_1_input
?
a
E__inference_UpSample_2_layer_call_and_return_conditional_losses_33456

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_34318

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
}
(__inference_Conv2D_1_layer_call_fn_34327

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_336352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?:
?
E__inference_sequential_layer_call_and_return_conditional_losses_33825

inputs
conv2d_1_33783
conv2d_1_33785
conv2d_2_33789
conv2d_2_33791
conv2d_3_33795
conv2d_3_33797
deconv2d_1_33802
deconv2d_1_33804
de2dtrans_2_33808
de2dtrans_2_33810
de2dtrans_3_33814
de2dtrans_3_33816
conv2dtrans_recon_33819
conv2dtrans_recon_33821
identity??)Conv2DTrans_recon/StatefulPartitionedCall? Conv2D_1/StatefulPartitionedCall? Conv2D_2/StatefulPartitionedCall? Conv2D_3/StatefulPartitionedCall?#De2DTrans_2/StatefulPartitionedCall?#De2DTrans_3/StatefulPartitionedCall?"DeConv2D_1/StatefulPartitionedCall?
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_33783conv2d_1_33785*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_336352"
 Conv2D_1/StatefulPartitionedCall?
MaxPool_1/PartitionedCallPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_1_layer_call_and_return_conditional_losses_333492
MaxPool_1/PartitionedCall?
 Conv2D_2/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_1/PartitionedCall:output:0conv2d_2_33789conv2d_2_33791*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_336632"
 Conv2D_2/StatefulPartitionedCall?
MaxPool_2/PartitionedCallPartitionedCall)Conv2D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_2_layer_call_and_return_conditional_losses_333612
MaxPool_2/PartitionedCall?
 Conv2D_3/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_2/PartitionedCall:output:0conv2d_3_33795conv2d_3_33797*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_336912"
 Conv2D_3/StatefulPartitionedCall?
MaxPool_3/PartitionedCallPartitionedCall)Conv2D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_3_layer_call_and_return_conditional_losses_333732
MaxPool_3/PartitionedCall?
UpSample_1/PartitionedCallPartitionedCall"MaxPool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_1_layer_call_and_return_conditional_losses_333922
UpSample_1/PartitionedCall?
"DeConv2D_1/StatefulPartitionedCallStatefulPartitionedCall#UpSample_1/PartitionedCall:output:0deconv2d_1_33802deconv2d_1_33804*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_DeConv2D_1_layer_call_and_return_conditional_losses_334332$
"DeConv2D_1/StatefulPartitionedCall?
UpSample_2/PartitionedCallPartitionedCall+DeConv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_2_layer_call_and_return_conditional_losses_334562
UpSample_2/PartitionedCall?
#De2DTrans_2/StatefulPartitionedCallStatefulPartitionedCall#UpSample_2/PartitionedCall:output:0de2dtrans_2_33808de2dtrans_2_33810*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_2_layer_call_and_return_conditional_losses_334972%
#De2DTrans_2/StatefulPartitionedCall?
UpSample_3/PartitionedCallPartitionedCall,De2DTrans_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_3_layer_call_and_return_conditional_losses_335202
UpSample_3/PartitionedCall?
#De2DTrans_3/StatefulPartitionedCallStatefulPartitionedCall#UpSample_3/PartitionedCall:output:0de2dtrans_3_33814de2dtrans_3_33816*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_3_layer_call_and_return_conditional_losses_335612%
#De2DTrans_3/StatefulPartitionedCall?
)Conv2DTrans_recon/StatefulPartitionedCallStatefulPartitionedCall,De2DTrans_3/StatefulPartitionedCall:output:0conv2dtrans_recon_33819conv2dtrans_recon_33821*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_Conv2DTrans_recon_layer_call_and_return_conditional_losses_336102+
)Conv2DTrans_recon/StatefulPartitionedCall?
IdentityIdentity2Conv2DTrans_recon/StatefulPartitionedCall:output:0*^Conv2DTrans_recon/StatefulPartitionedCall!^Conv2D_1/StatefulPartitionedCall!^Conv2D_2/StatefulPartitionedCall!^Conv2D_3/StatefulPartitionedCall$^De2DTrans_2/StatefulPartitionedCall$^De2DTrans_3/StatefulPartitionedCall#^DeConv2D_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2V
)Conv2DTrans_recon/StatefulPartitionedCall)Conv2DTrans_recon/StatefulPartitionedCall2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2D
 Conv2D_2/StatefulPartitionedCall Conv2D_2/StatefulPartitionedCall2D
 Conv2D_3/StatefulPartitionedCall Conv2D_3/StatefulPartitionedCall2J
#De2DTrans_2/StatefulPartitionedCall#De2DTrans_2/StatefulPartitionedCall2J
#De2DTrans_3/StatefulPartitionedCall#De2DTrans_3/StatefulPartitionedCall2H
"DeConv2D_1/StatefulPartitionedCall"DeConv2D_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_MaxPool_1_layer_call_fn_33355

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_MaxPool_1_layer_call_and_return_conditional_losses_333492
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

*__inference_DeConv2D_1_layer_call_fn_33443

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_DeConv2D_1_layer_call_and_return_conditional_losses_334332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_De2DTrans_2_layer_call_fn_33507

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_2_layer_call_and_return_conditional_losses_334972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_UpSample_2_layer_call_fn_33462

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_2_layer_call_and_return_conditional_losses_334562
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
1__inference_Conv2DTrans_recon_layer_call_fn_33620

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_Conv2DTrans_recon_layer_call_and_return_conditional_losses_336102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_33635

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
}
(__inference_Conv2D_3_layer_call_fn_34367

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_336912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?

?
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_34338

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
+__inference_De2DTrans_3_layer_call_fn_33571

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_De2DTrans_3_layer_call_and_return_conditional_losses_335612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_33663

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
F
*__inference_UpSample_1_layer_call_fn_33398

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_UpSample_1_layer_call_and_return_conditional_losses_333922
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?$
?
F__inference_De2DTrans_3_layer_call_and_return_conditional_losses_33561

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
¸
?

E__inference_sequential_layer_call_and_return_conditional_losses_34241

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource7
3deconv2d_1_conv2d_transpose_readvariableop_resource.
*deconv2d_1_biasadd_readvariableop_resource8
4de2dtrans_2_conv2d_transpose_readvariableop_resource/
+de2dtrans_2_biasadd_readvariableop_resource8
4de2dtrans_3_conv2d_transpose_readvariableop_resource/
+de2dtrans_3_biasadd_readvariableop_resource>
:conv2dtrans_recon_conv2d_transpose_readvariableop_resource5
1conv2dtrans_recon_biasadd_readvariableop_resource
identity??(Conv2DTrans_recon/BiasAdd/ReadVariableOp?1Conv2DTrans_recon/conv2d_transpose/ReadVariableOp?Conv2D_1/BiasAdd/ReadVariableOp?Conv2D_1/Conv2D/ReadVariableOp?Conv2D_2/BiasAdd/ReadVariableOp?Conv2D_2/Conv2D/ReadVariableOp?Conv2D_3/BiasAdd/ReadVariableOp?Conv2D_3/Conv2D/ReadVariableOp?"De2DTrans_2/BiasAdd/ReadVariableOp?+De2DTrans_2/conv2d_transpose/ReadVariableOp?"De2DTrans_3/BiasAdd/ReadVariableOp?+De2DTrans_3/conv2d_transpose/ReadVariableOp?!DeConv2D_1/BiasAdd/ReadVariableOp?*DeConv2D_1/conv2d_transpose/ReadVariableOp?
Conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
Conv2D_1/Conv2D/ReadVariableOp?
Conv2D_1/Conv2DConv2Dinputs&Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D_1/Conv2D?
Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv2D_1/BiasAdd/ReadVariableOp?
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2D:output:0'Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
Conv2D_1/BiasAdd}
Conv2D_1/ReluReluConv2D_1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Conv2D_1/Relu?
MaxPool_1/MaxPoolMaxPoolConv2D_1/Relu:activations:0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2
MaxPool_1/MaxPool?
Conv2D_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
Conv2D_2/Conv2D/ReadVariableOp?
Conv2D_2/Conv2DConv2DMaxPool_1/MaxPool:output:0&Conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D_2/Conv2D?
Conv2D_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
Conv2D_2/BiasAdd/ReadVariableOp?
Conv2D_2/BiasAddBiasAddConv2D_2/Conv2D:output:0'Conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
Conv2D_2/BiasAdd}
Conv2D_2/ReluReluConv2D_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Conv2D_2/Relu?
MaxPool_2/MaxPoolMaxPoolConv2D_2/Relu:activations:0*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
2
MaxPool_2/MaxPool?
Conv2D_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
Conv2D_3/Conv2D/ReadVariableOp?
Conv2D_3/Conv2DConv2DMaxPool_2/MaxPool:output:0&Conv2D_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
Conv2D_3/Conv2D?
Conv2D_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
Conv2D_3/BiasAdd/ReadVariableOp?
Conv2D_3/BiasAddBiasAddConv2D_3/Conv2D:output:0'Conv2D_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
Conv2D_3/BiasAdd|
Conv2D_3/ReluReluConv2D_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
Conv2D_3/Relu?
MaxPool_3/MaxPoolMaxPoolConv2D_3/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2
MaxPool_3/MaxPooln
UpSample_1/ShapeShapeMaxPool_3/MaxPool:output:0*
T0*
_output_shapes
:2
UpSample_1/Shape?
UpSample_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
UpSample_1/strided_slice/stack?
 UpSample_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_1/strided_slice/stack_1?
 UpSample_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_1/strided_slice/stack_2?
UpSample_1/strided_sliceStridedSliceUpSample_1/Shape:output:0'UpSample_1/strided_slice/stack:output:0)UpSample_1/strided_slice/stack_1:output:0)UpSample_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
UpSample_1/strided_sliceu
UpSample_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
UpSample_1/Const?
UpSample_1/mulMul!UpSample_1/strided_slice:output:0UpSample_1/Const:output:0*
T0*
_output_shapes
:2
UpSample_1/mul?
 UpSample_1/resize/ResizeBilinearResizeBilinearMaxPool_3/MaxPool:output:0UpSample_1/mul:z:0*
T0*0
_output_shapes
:?????????@@?*
half_pixel_centers(2"
 UpSample_1/resize/ResizeBilinear?
DeConv2D_1/ShapeShape1UpSample_1/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
DeConv2D_1/Shape?
DeConv2D_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
DeConv2D_1/strided_slice/stack?
 DeConv2D_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 DeConv2D_1/strided_slice/stack_1?
 DeConv2D_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 DeConv2D_1/strided_slice/stack_2?
DeConv2D_1/strided_sliceStridedSliceDeConv2D_1/Shape:output:0'DeConv2D_1/strided_slice/stack:output:0)DeConv2D_1/strided_slice/stack_1:output:0)DeConv2D_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
DeConv2D_1/strided_slicej
DeConv2D_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
DeConv2D_1/stack/1j
DeConv2D_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
DeConv2D_1/stack/2k
DeConv2D_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
DeConv2D_1/stack/3?
DeConv2D_1/stackPack!DeConv2D_1/strided_slice:output:0DeConv2D_1/stack/1:output:0DeConv2D_1/stack/2:output:0DeConv2D_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
DeConv2D_1/stack?
 DeConv2D_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 DeConv2D_1/strided_slice_1/stack?
"DeConv2D_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"DeConv2D_1/strided_slice_1/stack_1?
"DeConv2D_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"DeConv2D_1/strided_slice_1/stack_2?
DeConv2D_1/strided_slice_1StridedSliceDeConv2D_1/stack:output:0)DeConv2D_1/strided_slice_1/stack:output:0+DeConv2D_1/strided_slice_1/stack_1:output:0+DeConv2D_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
DeConv2D_1/strided_slice_1?
*DeConv2D_1/conv2d_transpose/ReadVariableOpReadVariableOp3deconv2d_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*DeConv2D_1/conv2d_transpose/ReadVariableOp?
DeConv2D_1/conv2d_transposeConv2DBackpropInputDeConv2D_1/stack:output:02DeConv2D_1/conv2d_transpose/ReadVariableOp:value:01UpSample_1/resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
DeConv2D_1/conv2d_transpose?
!DeConv2D_1/BiasAdd/ReadVariableOpReadVariableOp*deconv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!DeConv2D_1/BiasAdd/ReadVariableOp?
DeConv2D_1/BiasAddBiasAdd$DeConv2D_1/conv2d_transpose:output:0)DeConv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
DeConv2D_1/BiasAdd?
DeConv2D_1/ReluReluDeConv2D_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
DeConv2D_1/Reluq
UpSample_2/ShapeShapeDeConv2D_1/Relu:activations:0*
T0*
_output_shapes
:2
UpSample_2/Shape?
UpSample_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
UpSample_2/strided_slice/stack?
 UpSample_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_2/strided_slice/stack_1?
 UpSample_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_2/strided_slice/stack_2?
UpSample_2/strided_sliceStridedSliceUpSample_2/Shape:output:0'UpSample_2/strided_slice/stack:output:0)UpSample_2/strided_slice/stack_1:output:0)UpSample_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
UpSample_2/strided_sliceu
UpSample_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
UpSample_2/Const?
UpSample_2/mulMul!UpSample_2/strided_slice:output:0UpSample_2/Const:output:0*
T0*
_output_shapes
:2
UpSample_2/mul?
 UpSample_2/resize/ResizeBilinearResizeBilinearDeConv2D_1/Relu:activations:0UpSample_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2"
 UpSample_2/resize/ResizeBilinear?
De2DTrans_2/ShapeShape1UpSample_2/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
De2DTrans_2/Shape?
De2DTrans_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
De2DTrans_2/strided_slice/stack?
!De2DTrans_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!De2DTrans_2/strided_slice/stack_1?
!De2DTrans_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!De2DTrans_2/strided_slice/stack_2?
De2DTrans_2/strided_sliceStridedSliceDe2DTrans_2/Shape:output:0(De2DTrans_2/strided_slice/stack:output:0*De2DTrans_2/strided_slice/stack_1:output:0*De2DTrans_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
De2DTrans_2/strided_slicem
De2DTrans_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
De2DTrans_2/stack/1m
De2DTrans_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
De2DTrans_2/stack/2l
De2DTrans_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
De2DTrans_2/stack/3?
De2DTrans_2/stackPack"De2DTrans_2/strided_slice:output:0De2DTrans_2/stack/1:output:0De2DTrans_2/stack/2:output:0De2DTrans_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
De2DTrans_2/stack?
!De2DTrans_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!De2DTrans_2/strided_slice_1/stack?
#De2DTrans_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#De2DTrans_2/strided_slice_1/stack_1?
#De2DTrans_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#De2DTrans_2/strided_slice_1/stack_2?
De2DTrans_2/strided_slice_1StridedSliceDe2DTrans_2/stack:output:0*De2DTrans_2/strided_slice_1/stack:output:0,De2DTrans_2/strided_slice_1/stack_1:output:0,De2DTrans_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
De2DTrans_2/strided_slice_1?
+De2DTrans_2/conv2d_transpose/ReadVariableOpReadVariableOp4de2dtrans_2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+De2DTrans_2/conv2d_transpose/ReadVariableOp?
De2DTrans_2/conv2d_transposeConv2DBackpropInputDe2DTrans_2/stack:output:03De2DTrans_2/conv2d_transpose/ReadVariableOp:value:01UpSample_2/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
De2DTrans_2/conv2d_transpose?
"De2DTrans_2/BiasAdd/ReadVariableOpReadVariableOp+de2dtrans_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"De2DTrans_2/BiasAdd/ReadVariableOp?
De2DTrans_2/BiasAddBiasAdd%De2DTrans_2/conv2d_transpose:output:0*De2DTrans_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
De2DTrans_2/BiasAdd?
De2DTrans_2/ReluReluDe2DTrans_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
De2DTrans_2/Relur
UpSample_3/ShapeShapeDe2DTrans_2/Relu:activations:0*
T0*
_output_shapes
:2
UpSample_3/Shape?
UpSample_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
UpSample_3/strided_slice/stack?
 UpSample_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_3/strided_slice/stack_1?
 UpSample_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_3/strided_slice/stack_2?
UpSample_3/strided_sliceStridedSliceUpSample_3/Shape:output:0'UpSample_3/strided_slice/stack:output:0)UpSample_3/strided_slice/stack_1:output:0)UpSample_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
UpSample_3/strided_sliceu
UpSample_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
UpSample_3/Const?
UpSample_3/mulMul!UpSample_3/strided_slice:output:0UpSample_3/Const:output:0*
T0*
_output_shapes
:2
UpSample_3/mul?
 UpSample_3/resize/ResizeBilinearResizeBilinearDe2DTrans_2/Relu:activations:0UpSample_3/mul:z:0*
T0*1
_output_shapes
:???????????@*
half_pixel_centers(2"
 UpSample_3/resize/ResizeBilinear?
De2DTrans_3/ShapeShape1UpSample_3/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
De2DTrans_3/Shape?
De2DTrans_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
De2DTrans_3/strided_slice/stack?
!De2DTrans_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!De2DTrans_3/strided_slice/stack_1?
!De2DTrans_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!De2DTrans_3/strided_slice/stack_2?
De2DTrans_3/strided_sliceStridedSliceDe2DTrans_3/Shape:output:0(De2DTrans_3/strided_slice/stack:output:0*De2DTrans_3/strided_slice/stack_1:output:0*De2DTrans_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
De2DTrans_3/strided_slicem
De2DTrans_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
De2DTrans_3/stack/1m
De2DTrans_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
De2DTrans_3/stack/2l
De2DTrans_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
De2DTrans_3/stack/3?
De2DTrans_3/stackPack"De2DTrans_3/strided_slice:output:0De2DTrans_3/stack/1:output:0De2DTrans_3/stack/2:output:0De2DTrans_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
De2DTrans_3/stack?
!De2DTrans_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!De2DTrans_3/strided_slice_1/stack?
#De2DTrans_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#De2DTrans_3/strided_slice_1/stack_1?
#De2DTrans_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#De2DTrans_3/strided_slice_1/stack_2?
De2DTrans_3/strided_slice_1StridedSliceDe2DTrans_3/stack:output:0*De2DTrans_3/strided_slice_1/stack:output:0,De2DTrans_3/strided_slice_1/stack_1:output:0,De2DTrans_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
De2DTrans_3/strided_slice_1?
+De2DTrans_3/conv2d_transpose/ReadVariableOpReadVariableOp4de2dtrans_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+De2DTrans_3/conv2d_transpose/ReadVariableOp?
De2DTrans_3/conv2d_transposeConv2DBackpropInputDe2DTrans_3/stack:output:03De2DTrans_3/conv2d_transpose/ReadVariableOp:value:01UpSample_3/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
De2DTrans_3/conv2d_transpose?
"De2DTrans_3/BiasAdd/ReadVariableOpReadVariableOp+de2dtrans_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"De2DTrans_3/BiasAdd/ReadVariableOp?
De2DTrans_3/BiasAddBiasAdd%De2DTrans_3/conv2d_transpose:output:0*De2DTrans_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
De2DTrans_3/BiasAdd?
De2DTrans_3/ReluReluDe2DTrans_3/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
De2DTrans_3/Relu?
Conv2DTrans_recon/ShapeShapeDe2DTrans_3/Relu:activations:0*
T0*
_output_shapes
:2
Conv2DTrans_recon/Shape?
%Conv2DTrans_recon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Conv2DTrans_recon/strided_slice/stack?
'Conv2DTrans_recon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv2DTrans_recon/strided_slice/stack_1?
'Conv2DTrans_recon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv2DTrans_recon/strided_slice/stack_2?
Conv2DTrans_recon/strided_sliceStridedSlice Conv2DTrans_recon/Shape:output:0.Conv2DTrans_recon/strided_slice/stack:output:00Conv2DTrans_recon/strided_slice/stack_1:output:00Conv2DTrans_recon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Conv2DTrans_recon/strided_slicey
Conv2DTrans_recon/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Conv2DTrans_recon/stack/1y
Conv2DTrans_recon/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Conv2DTrans_recon/stack/2x
Conv2DTrans_recon/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Conv2DTrans_recon/stack/3?
Conv2DTrans_recon/stackPack(Conv2DTrans_recon/strided_slice:output:0"Conv2DTrans_recon/stack/1:output:0"Conv2DTrans_recon/stack/2:output:0"Conv2DTrans_recon/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv2DTrans_recon/stack?
'Conv2DTrans_recon/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Conv2DTrans_recon/strided_slice_1/stack?
)Conv2DTrans_recon/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Conv2DTrans_recon/strided_slice_1/stack_1?
)Conv2DTrans_recon/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Conv2DTrans_recon/strided_slice_1/stack_2?
!Conv2DTrans_recon/strided_slice_1StridedSlice Conv2DTrans_recon/stack:output:00Conv2DTrans_recon/strided_slice_1/stack:output:02Conv2DTrans_recon/strided_slice_1/stack_1:output:02Conv2DTrans_recon/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Conv2DTrans_recon/strided_slice_1?
1Conv2DTrans_recon/conv2d_transpose/ReadVariableOpReadVariableOp:conv2dtrans_recon_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype023
1Conv2DTrans_recon/conv2d_transpose/ReadVariableOp?
"Conv2DTrans_recon/conv2d_transposeConv2DBackpropInput Conv2DTrans_recon/stack:output:09Conv2DTrans_recon/conv2d_transpose/ReadVariableOp:value:0De2DTrans_3/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2$
"Conv2DTrans_recon/conv2d_transpose?
(Conv2DTrans_recon/BiasAdd/ReadVariableOpReadVariableOp1conv2dtrans_recon_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Conv2DTrans_recon/BiasAdd/ReadVariableOp?
Conv2DTrans_recon/BiasAddBiasAdd+Conv2DTrans_recon/conv2d_transpose:output:00Conv2DTrans_recon/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
Conv2DTrans_recon/BiasAdd?
Conv2DTrans_recon/ReluRelu"Conv2DTrans_recon/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Conv2DTrans_recon/Relu?
IdentityIdentity$Conv2DTrans_recon/Relu:activations:0)^Conv2DTrans_recon/BiasAdd/ReadVariableOp2^Conv2DTrans_recon/conv2d_transpose/ReadVariableOp ^Conv2D_1/BiasAdd/ReadVariableOp^Conv2D_1/Conv2D/ReadVariableOp ^Conv2D_2/BiasAdd/ReadVariableOp^Conv2D_2/Conv2D/ReadVariableOp ^Conv2D_3/BiasAdd/ReadVariableOp^Conv2D_3/Conv2D/ReadVariableOp#^De2DTrans_2/BiasAdd/ReadVariableOp,^De2DTrans_2/conv2d_transpose/ReadVariableOp#^De2DTrans_3/BiasAdd/ReadVariableOp,^De2DTrans_3/conv2d_transpose/ReadVariableOp"^DeConv2D_1/BiasAdd/ReadVariableOp+^DeConv2D_1/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2T
(Conv2DTrans_recon/BiasAdd/ReadVariableOp(Conv2DTrans_recon/BiasAdd/ReadVariableOp2f
1Conv2DTrans_recon/conv2d_transpose/ReadVariableOp1Conv2DTrans_recon/conv2d_transpose/ReadVariableOp2B
Conv2D_1/BiasAdd/ReadVariableOpConv2D_1/BiasAdd/ReadVariableOp2@
Conv2D_1/Conv2D/ReadVariableOpConv2D_1/Conv2D/ReadVariableOp2B
Conv2D_2/BiasAdd/ReadVariableOpConv2D_2/BiasAdd/ReadVariableOp2@
Conv2D_2/Conv2D/ReadVariableOpConv2D_2/Conv2D/ReadVariableOp2B
Conv2D_3/BiasAdd/ReadVariableOpConv2D_3/BiasAdd/ReadVariableOp2@
Conv2D_3/Conv2D/ReadVariableOpConv2D_3/Conv2D/ReadVariableOp2H
"De2DTrans_2/BiasAdd/ReadVariableOp"De2DTrans_2/BiasAdd/ReadVariableOp2Z
+De2DTrans_2/conv2d_transpose/ReadVariableOp+De2DTrans_2/conv2d_transpose/ReadVariableOp2H
"De2DTrans_3/BiasAdd/ReadVariableOp"De2DTrans_3/BiasAdd/ReadVariableOp2Z
+De2DTrans_3/conv2d_transpose/ReadVariableOp+De2DTrans_3/conv2d_transpose/ReadVariableOp2F
!DeConv2D_1/BiasAdd/ReadVariableOp!DeConv2D_1/BiasAdd/ReadVariableOp2X
*DeConv2D_1/conv2d_transpose/ReadVariableOp*DeConv2D_1/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_MaxPool_2_layer_call_and_return_conditional_losses_33361

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
¸
?

E__inference_sequential_layer_call_and_return_conditional_losses_34109

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource7
3deconv2d_1_conv2d_transpose_readvariableop_resource.
*deconv2d_1_biasadd_readvariableop_resource8
4de2dtrans_2_conv2d_transpose_readvariableop_resource/
+de2dtrans_2_biasadd_readvariableop_resource8
4de2dtrans_3_conv2d_transpose_readvariableop_resource/
+de2dtrans_3_biasadd_readvariableop_resource>
:conv2dtrans_recon_conv2d_transpose_readvariableop_resource5
1conv2dtrans_recon_biasadd_readvariableop_resource
identity??(Conv2DTrans_recon/BiasAdd/ReadVariableOp?1Conv2DTrans_recon/conv2d_transpose/ReadVariableOp?Conv2D_1/BiasAdd/ReadVariableOp?Conv2D_1/Conv2D/ReadVariableOp?Conv2D_2/BiasAdd/ReadVariableOp?Conv2D_2/Conv2D/ReadVariableOp?Conv2D_3/BiasAdd/ReadVariableOp?Conv2D_3/Conv2D/ReadVariableOp?"De2DTrans_2/BiasAdd/ReadVariableOp?+De2DTrans_2/conv2d_transpose/ReadVariableOp?"De2DTrans_3/BiasAdd/ReadVariableOp?+De2DTrans_3/conv2d_transpose/ReadVariableOp?!DeConv2D_1/BiasAdd/ReadVariableOp?*DeConv2D_1/conv2d_transpose/ReadVariableOp?
Conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
Conv2D_1/Conv2D/ReadVariableOp?
Conv2D_1/Conv2DConv2Dinputs&Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D_1/Conv2D?
Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv2D_1/BiasAdd/ReadVariableOp?
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2D:output:0'Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
Conv2D_1/BiasAdd}
Conv2D_1/ReluReluConv2D_1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Conv2D_1/Relu?
MaxPool_1/MaxPoolMaxPoolConv2D_1/Relu:activations:0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2
MaxPool_1/MaxPool?
Conv2D_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
Conv2D_2/Conv2D/ReadVariableOp?
Conv2D_2/Conv2DConv2DMaxPool_1/MaxPool:output:0&Conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D_2/Conv2D?
Conv2D_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
Conv2D_2/BiasAdd/ReadVariableOp?
Conv2D_2/BiasAddBiasAddConv2D_2/Conv2D:output:0'Conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
Conv2D_2/BiasAdd}
Conv2D_2/ReluReluConv2D_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Conv2D_2/Relu?
MaxPool_2/MaxPoolMaxPoolConv2D_2/Relu:activations:0*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
2
MaxPool_2/MaxPool?
Conv2D_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
Conv2D_3/Conv2D/ReadVariableOp?
Conv2D_3/Conv2DConv2DMaxPool_2/MaxPool:output:0&Conv2D_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
Conv2D_3/Conv2D?
Conv2D_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
Conv2D_3/BiasAdd/ReadVariableOp?
Conv2D_3/BiasAddBiasAddConv2D_3/Conv2D:output:0'Conv2D_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
Conv2D_3/BiasAdd|
Conv2D_3/ReluReluConv2D_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
Conv2D_3/Relu?
MaxPool_3/MaxPoolMaxPoolConv2D_3/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2
MaxPool_3/MaxPooln
UpSample_1/ShapeShapeMaxPool_3/MaxPool:output:0*
T0*
_output_shapes
:2
UpSample_1/Shape?
UpSample_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
UpSample_1/strided_slice/stack?
 UpSample_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_1/strided_slice/stack_1?
 UpSample_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_1/strided_slice/stack_2?
UpSample_1/strided_sliceStridedSliceUpSample_1/Shape:output:0'UpSample_1/strided_slice/stack:output:0)UpSample_1/strided_slice/stack_1:output:0)UpSample_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
UpSample_1/strided_sliceu
UpSample_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
UpSample_1/Const?
UpSample_1/mulMul!UpSample_1/strided_slice:output:0UpSample_1/Const:output:0*
T0*
_output_shapes
:2
UpSample_1/mul?
 UpSample_1/resize/ResizeBilinearResizeBilinearMaxPool_3/MaxPool:output:0UpSample_1/mul:z:0*
T0*0
_output_shapes
:?????????@@?*
half_pixel_centers(2"
 UpSample_1/resize/ResizeBilinear?
DeConv2D_1/ShapeShape1UpSample_1/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
DeConv2D_1/Shape?
DeConv2D_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
DeConv2D_1/strided_slice/stack?
 DeConv2D_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 DeConv2D_1/strided_slice/stack_1?
 DeConv2D_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 DeConv2D_1/strided_slice/stack_2?
DeConv2D_1/strided_sliceStridedSliceDeConv2D_1/Shape:output:0'DeConv2D_1/strided_slice/stack:output:0)DeConv2D_1/strided_slice/stack_1:output:0)DeConv2D_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
DeConv2D_1/strided_slicej
DeConv2D_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
DeConv2D_1/stack/1j
DeConv2D_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
DeConv2D_1/stack/2k
DeConv2D_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
DeConv2D_1/stack/3?
DeConv2D_1/stackPack!DeConv2D_1/strided_slice:output:0DeConv2D_1/stack/1:output:0DeConv2D_1/stack/2:output:0DeConv2D_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
DeConv2D_1/stack?
 DeConv2D_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 DeConv2D_1/strided_slice_1/stack?
"DeConv2D_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"DeConv2D_1/strided_slice_1/stack_1?
"DeConv2D_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"DeConv2D_1/strided_slice_1/stack_2?
DeConv2D_1/strided_slice_1StridedSliceDeConv2D_1/stack:output:0)DeConv2D_1/strided_slice_1/stack:output:0+DeConv2D_1/strided_slice_1/stack_1:output:0+DeConv2D_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
DeConv2D_1/strided_slice_1?
*DeConv2D_1/conv2d_transpose/ReadVariableOpReadVariableOp3deconv2d_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*DeConv2D_1/conv2d_transpose/ReadVariableOp?
DeConv2D_1/conv2d_transposeConv2DBackpropInputDeConv2D_1/stack:output:02DeConv2D_1/conv2d_transpose/ReadVariableOp:value:01UpSample_1/resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
DeConv2D_1/conv2d_transpose?
!DeConv2D_1/BiasAdd/ReadVariableOpReadVariableOp*deconv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!DeConv2D_1/BiasAdd/ReadVariableOp?
DeConv2D_1/BiasAddBiasAdd$DeConv2D_1/conv2d_transpose:output:0)DeConv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
DeConv2D_1/BiasAdd?
DeConv2D_1/ReluReluDeConv2D_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
DeConv2D_1/Reluq
UpSample_2/ShapeShapeDeConv2D_1/Relu:activations:0*
T0*
_output_shapes
:2
UpSample_2/Shape?
UpSample_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
UpSample_2/strided_slice/stack?
 UpSample_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_2/strided_slice/stack_1?
 UpSample_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_2/strided_slice/stack_2?
UpSample_2/strided_sliceStridedSliceUpSample_2/Shape:output:0'UpSample_2/strided_slice/stack:output:0)UpSample_2/strided_slice/stack_1:output:0)UpSample_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
UpSample_2/strided_sliceu
UpSample_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
UpSample_2/Const?
UpSample_2/mulMul!UpSample_2/strided_slice:output:0UpSample_2/Const:output:0*
T0*
_output_shapes
:2
UpSample_2/mul?
 UpSample_2/resize/ResizeBilinearResizeBilinearDeConv2D_1/Relu:activations:0UpSample_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2"
 UpSample_2/resize/ResizeBilinear?
De2DTrans_2/ShapeShape1UpSample_2/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
De2DTrans_2/Shape?
De2DTrans_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
De2DTrans_2/strided_slice/stack?
!De2DTrans_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!De2DTrans_2/strided_slice/stack_1?
!De2DTrans_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!De2DTrans_2/strided_slice/stack_2?
De2DTrans_2/strided_sliceStridedSliceDe2DTrans_2/Shape:output:0(De2DTrans_2/strided_slice/stack:output:0*De2DTrans_2/strided_slice/stack_1:output:0*De2DTrans_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
De2DTrans_2/strided_slicem
De2DTrans_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
De2DTrans_2/stack/1m
De2DTrans_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
De2DTrans_2/stack/2l
De2DTrans_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
De2DTrans_2/stack/3?
De2DTrans_2/stackPack"De2DTrans_2/strided_slice:output:0De2DTrans_2/stack/1:output:0De2DTrans_2/stack/2:output:0De2DTrans_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
De2DTrans_2/stack?
!De2DTrans_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!De2DTrans_2/strided_slice_1/stack?
#De2DTrans_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#De2DTrans_2/strided_slice_1/stack_1?
#De2DTrans_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#De2DTrans_2/strided_slice_1/stack_2?
De2DTrans_2/strided_slice_1StridedSliceDe2DTrans_2/stack:output:0*De2DTrans_2/strided_slice_1/stack:output:0,De2DTrans_2/strided_slice_1/stack_1:output:0,De2DTrans_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
De2DTrans_2/strided_slice_1?
+De2DTrans_2/conv2d_transpose/ReadVariableOpReadVariableOp4de2dtrans_2_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+De2DTrans_2/conv2d_transpose/ReadVariableOp?
De2DTrans_2/conv2d_transposeConv2DBackpropInputDe2DTrans_2/stack:output:03De2DTrans_2/conv2d_transpose/ReadVariableOp:value:01UpSample_2/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
De2DTrans_2/conv2d_transpose?
"De2DTrans_2/BiasAdd/ReadVariableOpReadVariableOp+de2dtrans_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"De2DTrans_2/BiasAdd/ReadVariableOp?
De2DTrans_2/BiasAddBiasAdd%De2DTrans_2/conv2d_transpose:output:0*De2DTrans_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
De2DTrans_2/BiasAdd?
De2DTrans_2/ReluReluDe2DTrans_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
De2DTrans_2/Relur
UpSample_3/ShapeShapeDe2DTrans_2/Relu:activations:0*
T0*
_output_shapes
:2
UpSample_3/Shape?
UpSample_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
UpSample_3/strided_slice/stack?
 UpSample_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_3/strided_slice/stack_1?
 UpSample_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 UpSample_3/strided_slice/stack_2?
UpSample_3/strided_sliceStridedSliceUpSample_3/Shape:output:0'UpSample_3/strided_slice/stack:output:0)UpSample_3/strided_slice/stack_1:output:0)UpSample_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
UpSample_3/strided_sliceu
UpSample_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
UpSample_3/Const?
UpSample_3/mulMul!UpSample_3/strided_slice:output:0UpSample_3/Const:output:0*
T0*
_output_shapes
:2
UpSample_3/mul?
 UpSample_3/resize/ResizeBilinearResizeBilinearDe2DTrans_2/Relu:activations:0UpSample_3/mul:z:0*
T0*1
_output_shapes
:???????????@*
half_pixel_centers(2"
 UpSample_3/resize/ResizeBilinear?
De2DTrans_3/ShapeShape1UpSample_3/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:2
De2DTrans_3/Shape?
De2DTrans_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
De2DTrans_3/strided_slice/stack?
!De2DTrans_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!De2DTrans_3/strided_slice/stack_1?
!De2DTrans_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!De2DTrans_3/strided_slice/stack_2?
De2DTrans_3/strided_sliceStridedSliceDe2DTrans_3/Shape:output:0(De2DTrans_3/strided_slice/stack:output:0*De2DTrans_3/strided_slice/stack_1:output:0*De2DTrans_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
De2DTrans_3/strided_slicem
De2DTrans_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
De2DTrans_3/stack/1m
De2DTrans_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
De2DTrans_3/stack/2l
De2DTrans_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
De2DTrans_3/stack/3?
De2DTrans_3/stackPack"De2DTrans_3/strided_slice:output:0De2DTrans_3/stack/1:output:0De2DTrans_3/stack/2:output:0De2DTrans_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
De2DTrans_3/stack?
!De2DTrans_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!De2DTrans_3/strided_slice_1/stack?
#De2DTrans_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#De2DTrans_3/strided_slice_1/stack_1?
#De2DTrans_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#De2DTrans_3/strided_slice_1/stack_2?
De2DTrans_3/strided_slice_1StridedSliceDe2DTrans_3/stack:output:0*De2DTrans_3/strided_slice_1/stack:output:0,De2DTrans_3/strided_slice_1/stack_1:output:0,De2DTrans_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
De2DTrans_3/strided_slice_1?
+De2DTrans_3/conv2d_transpose/ReadVariableOpReadVariableOp4de2dtrans_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+De2DTrans_3/conv2d_transpose/ReadVariableOp?
De2DTrans_3/conv2d_transposeConv2DBackpropInputDe2DTrans_3/stack:output:03De2DTrans_3/conv2d_transpose/ReadVariableOp:value:01UpSample_3/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
De2DTrans_3/conv2d_transpose?
"De2DTrans_3/BiasAdd/ReadVariableOpReadVariableOp+de2dtrans_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"De2DTrans_3/BiasAdd/ReadVariableOp?
De2DTrans_3/BiasAddBiasAdd%De2DTrans_3/conv2d_transpose:output:0*De2DTrans_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
De2DTrans_3/BiasAdd?
De2DTrans_3/ReluReluDe2DTrans_3/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
De2DTrans_3/Relu?
Conv2DTrans_recon/ShapeShapeDe2DTrans_3/Relu:activations:0*
T0*
_output_shapes
:2
Conv2DTrans_recon/Shape?
%Conv2DTrans_recon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Conv2DTrans_recon/strided_slice/stack?
'Conv2DTrans_recon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv2DTrans_recon/strided_slice/stack_1?
'Conv2DTrans_recon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv2DTrans_recon/strided_slice/stack_2?
Conv2DTrans_recon/strided_sliceStridedSlice Conv2DTrans_recon/Shape:output:0.Conv2DTrans_recon/strided_slice/stack:output:00Conv2DTrans_recon/strided_slice/stack_1:output:00Conv2DTrans_recon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Conv2DTrans_recon/strided_slicey
Conv2DTrans_recon/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Conv2DTrans_recon/stack/1y
Conv2DTrans_recon/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Conv2DTrans_recon/stack/2x
Conv2DTrans_recon/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Conv2DTrans_recon/stack/3?
Conv2DTrans_recon/stackPack(Conv2DTrans_recon/strided_slice:output:0"Conv2DTrans_recon/stack/1:output:0"Conv2DTrans_recon/stack/2:output:0"Conv2DTrans_recon/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv2DTrans_recon/stack?
'Conv2DTrans_recon/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Conv2DTrans_recon/strided_slice_1/stack?
)Conv2DTrans_recon/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Conv2DTrans_recon/strided_slice_1/stack_1?
)Conv2DTrans_recon/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Conv2DTrans_recon/strided_slice_1/stack_2?
!Conv2DTrans_recon/strided_slice_1StridedSlice Conv2DTrans_recon/stack:output:00Conv2DTrans_recon/strided_slice_1/stack:output:02Conv2DTrans_recon/strided_slice_1/stack_1:output:02Conv2DTrans_recon/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Conv2DTrans_recon/strided_slice_1?
1Conv2DTrans_recon/conv2d_transpose/ReadVariableOpReadVariableOp:conv2dtrans_recon_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype023
1Conv2DTrans_recon/conv2d_transpose/ReadVariableOp?
"Conv2DTrans_recon/conv2d_transposeConv2DBackpropInput Conv2DTrans_recon/stack:output:09Conv2DTrans_recon/conv2d_transpose/ReadVariableOp:value:0De2DTrans_3/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2$
"Conv2DTrans_recon/conv2d_transpose?
(Conv2DTrans_recon/BiasAdd/ReadVariableOpReadVariableOp1conv2dtrans_recon_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Conv2DTrans_recon/BiasAdd/ReadVariableOp?
Conv2DTrans_recon/BiasAddBiasAdd+Conv2DTrans_recon/conv2d_transpose:output:00Conv2DTrans_recon/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
Conv2DTrans_recon/BiasAdd?
Conv2DTrans_recon/ReluRelu"Conv2DTrans_recon/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Conv2DTrans_recon/Relu?
IdentityIdentity$Conv2DTrans_recon/Relu:activations:0)^Conv2DTrans_recon/BiasAdd/ReadVariableOp2^Conv2DTrans_recon/conv2d_transpose/ReadVariableOp ^Conv2D_1/BiasAdd/ReadVariableOp^Conv2D_1/Conv2D/ReadVariableOp ^Conv2D_2/BiasAdd/ReadVariableOp^Conv2D_2/Conv2D/ReadVariableOp ^Conv2D_3/BiasAdd/ReadVariableOp^Conv2D_3/Conv2D/ReadVariableOp#^De2DTrans_2/BiasAdd/ReadVariableOp,^De2DTrans_2/conv2d_transpose/ReadVariableOp#^De2DTrans_3/BiasAdd/ReadVariableOp,^De2DTrans_3/conv2d_transpose/ReadVariableOp"^DeConv2D_1/BiasAdd/ReadVariableOp+^DeConv2D_1/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2T
(Conv2DTrans_recon/BiasAdd/ReadVariableOp(Conv2DTrans_recon/BiasAdd/ReadVariableOp2f
1Conv2DTrans_recon/conv2d_transpose/ReadVariableOp1Conv2DTrans_recon/conv2d_transpose/ReadVariableOp2B
Conv2D_1/BiasAdd/ReadVariableOpConv2D_1/BiasAdd/ReadVariableOp2@
Conv2D_1/Conv2D/ReadVariableOpConv2D_1/Conv2D/ReadVariableOp2B
Conv2D_2/BiasAdd/ReadVariableOpConv2D_2/BiasAdd/ReadVariableOp2@
Conv2D_2/Conv2D/ReadVariableOpConv2D_2/Conv2D/ReadVariableOp2B
Conv2D_3/BiasAdd/ReadVariableOpConv2D_3/BiasAdd/ReadVariableOp2@
Conv2D_3/Conv2D/ReadVariableOpConv2D_3/Conv2D/ReadVariableOp2H
"De2DTrans_2/BiasAdd/ReadVariableOp"De2DTrans_2/BiasAdd/ReadVariableOp2Z
+De2DTrans_2/conv2d_transpose/ReadVariableOp+De2DTrans_2/conv2d_transpose/ReadVariableOp2H
"De2DTrans_3/BiasAdd/ReadVariableOp"De2DTrans_3/BiasAdd/ReadVariableOp2Z
+De2DTrans_3/conv2d_transpose/ReadVariableOp+De2DTrans_3/conv2d_transpose/ReadVariableOp2F
!DeConv2D_1/BiasAdd/ReadVariableOp!DeConv2D_1/BiasAdd/ReadVariableOp2X
*DeConv2D_1/conv2d_transpose/ReadVariableOp*DeConv2D_1/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
}
(__inference_Conv2D_2_layer_call_fn_34347

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_336632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?

?
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_34358

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
Conv2D_1_inputA
 serving_default_Conv2D_1_input:0???????????O
Conv2DTrans_recon:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:µ
?t
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?p
_tf_keras_sequential?p{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Conv2D_1_input"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D_1", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool_1", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool_2", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D_3", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool_3", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "UpSampling2D", "config": {"name": "UpSample_1", "trainable": false, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}}, {"class_name": "Conv2DTranspose", "config": {"name": "DeConv2D_1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "UpSample_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}}, {"class_name": "Conv2DTranspose", "config": {"name": "De2DTrans_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "UpSample_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}}, {"class_name": "Conv2DTranspose", "config": {"name": "De2DTrans_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv2DTrans_recon", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Conv2D_1_input"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D_1", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool_1", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool_2", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D_3", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool_3", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "UpSampling2D", "config": {"name": "UpSample_1", "trainable": false, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}}, {"class_name": "Conv2DTranspose", "config": {"name": "DeConv2D_1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "UpSample_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}}, {"class_name": "Conv2DTranspose", "config": {"name": "De2DTrans_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "UpSample_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}}, {"class_name": "Conv2DTranspose", "config": {"name": "De2DTrans_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv2DTrans_recon", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [[{"class_name": "Accuracy", "config": {"name": "accuracy", "dtype": "float32"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "Conv2D_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D_1", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "MaxPool_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MaxPool_1", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "Conv2D_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}
?
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "MaxPool_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MaxPool_2", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "Conv2D_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D_3", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
?
.regularization_losses
/trainable_variables
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "MaxPool_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MaxPool_3", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
2regularization_losses
3trainable_variables
4	variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "UpSample_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "UpSample_1", "trainable": false, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "DeConv2D_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "DeConv2D_1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}
?
<regularization_losses
=trainable_variables
>	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "UpSample_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "UpSample_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "De2DTrans_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "De2DTrans_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 128]}}
?
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "UpSample_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "UpSample_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "De2DTrans_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "De2DTrans_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 64]}}
?


Pkernel
Qbias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "Conv2DTrans_recon", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2DTrans_recon", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}
?
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_ratem?m?m?m?(m?)m?6m?7m?@m?Am?Jm?Km?Pm?Qm?v?v?v?v?(v?)v?6v?7v?@v?Av?Jv?Kv?Pv?Qv?"
	optimizer
 "
trackable_list_wrapper
J
@0
A1
J2
K3
P4
Q5"
trackable_list_wrapper
?
0
1
2
3
(4
)5
66
77
@8
A9
J10
K11
P12
Q13"
trackable_list_wrapper
?
[layer_metrics
regularization_losses
trainable_variables

\layers
	variables
]non_trainable_variables
^layer_regularization_losses
_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):' 2Conv2D_1/kernel
: 2Conv2D_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
`layer_metrics
regularization_losses
trainable_variables

alayers
	variables
bnon_trainable_variables
clayer_regularization_losses
dmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
elayer_metrics
regularization_losses
trainable_variables

flayers
	variables
gnon_trainable_variables
hlayer_regularization_losses
imetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' @2Conv2D_2/kernel
:@2Conv2D_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
jlayer_metrics
 regularization_losses
!trainable_variables

klayers
"	variables
lnon_trainable_variables
mlayer_regularization_losses
nmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
olayer_metrics
$regularization_losses
%trainable_variables

players
&	variables
qnon_trainable_variables
rlayer_regularization_losses
smetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@?2Conv2D_3/kernel
:?2Conv2D_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
tlayer_metrics
*regularization_losses
+trainable_variables

ulayers
,	variables
vnon_trainable_variables
wlayer_regularization_losses
xmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ylayer_metrics
.regularization_losses
/trainable_variables

zlayers
0	variables
{non_trainable_variables
|layer_regularization_losses
}metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
~layer_metrics
2regularization_losses
3trainable_variables

layers
4	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+??2DeConv2D_1/kernel
:?2DeConv2D_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
?layer_metrics
8regularization_losses
9trainable_variables
?layers
:	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
<regularization_losses
=trainable_variables
?layers
>	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+@?2De2DTrans_2/kernel
:@2De2DTrans_2/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
?layer_metrics
Bregularization_losses
Ctrainable_variables
?layers
D	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Fregularization_losses
Gtrainable_variables
?layers
H	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* @2De2DTrans_3/kernel
: 2De2DTrans_3/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
?layer_metrics
Lregularization_losses
Mtrainable_variables
?layers
N	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:0 2Conv2DTrans_recon/kernel
$:"2Conv2DTrans_recon/bias
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?layer_metrics
Rregularization_losses
Strainable_variables
?layers
T	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
X
0
1
2
3
(4
)5
66
77"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metricv{"class_name": "Accuracy", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
.:, 2Adam/Conv2D_1/kernel/m
 : 2Adam/Conv2D_1/bias/m
.:, @2Adam/Conv2D_2/kernel/m
 :@2Adam/Conv2D_2/bias/m
/:-@?2Adam/Conv2D_3/kernel/m
!:?2Adam/Conv2D_3/bias/m
2:0??2Adam/DeConv2D_1/kernel/m
#:!?2Adam/DeConv2D_1/bias/m
2:0@?2Adam/De2DTrans_2/kernel/m
#:!@2Adam/De2DTrans_2/bias/m
1:/ @2Adam/De2DTrans_3/kernel/m
#:! 2Adam/De2DTrans_3/bias/m
7:5 2Adam/Conv2DTrans_recon/kernel/m
):'2Adam/Conv2DTrans_recon/bias/m
.:, 2Adam/Conv2D_1/kernel/v
 : 2Adam/Conv2D_1/bias/v
.:, @2Adam/Conv2D_2/kernel/v
 :@2Adam/Conv2D_2/bias/v
/:-@?2Adam/Conv2D_3/kernel/v
!:?2Adam/Conv2D_3/bias/v
2:0??2Adam/DeConv2D_1/kernel/v
#:!?2Adam/DeConv2D_1/bias/v
2:0@?2Adam/De2DTrans_2/kernel/v
#:!@2Adam/De2DTrans_2/bias/v
1:/ @2Adam/De2DTrans_3/kernel/v
#:! 2Adam/De2DTrans_3/bias/v
7:5 2Adam/Conv2DTrans_recon/kernel/v
):'2Adam/Conv2DTrans_recon/bias/v
?2?
 __inference__wrapped_model_33343?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/
Conv2D_1_input???????????
?2?
*__inference_sequential_layer_call_fn_34274
*__inference_sequential_layer_call_fn_34307
*__inference_sequential_layer_call_fn_33934
*__inference_sequential_layer_call_fn_33856?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_34241
E__inference_sequential_layer_call_and_return_conditional_losses_33732
E__inference_sequential_layer_call_and_return_conditional_losses_34109
E__inference_sequential_layer_call_and_return_conditional_losses_33777?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_Conv2D_1_layer_call_fn_34327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_34318?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_MaxPool_1_layer_call_fn_33355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_MaxPool_1_layer_call_and_return_conditional_losses_33349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_Conv2D_2_layer_call_fn_34347?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_34338?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_MaxPool_2_layer_call_fn_33367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_MaxPool_2_layer_call_and_return_conditional_losses_33361?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_Conv2D_3_layer_call_fn_34367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_34358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_MaxPool_3_layer_call_fn_33379?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_MaxPool_3_layer_call_and_return_conditional_losses_33373?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_UpSample_1_layer_call_fn_33398?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_UpSample_1_layer_call_and_return_conditional_losses_33392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_DeConv2D_1_layer_call_fn_33443?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
E__inference_DeConv2D_1_layer_call_and_return_conditional_losses_33433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
*__inference_UpSample_2_layer_call_fn_33462?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_UpSample_2_layer_call_and_return_conditional_losses_33456?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_De2DTrans_2_layer_call_fn_33507?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
F__inference_De2DTrans_2_layer_call_and_return_conditional_losses_33497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
*__inference_UpSample_3_layer_call_fn_33526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_UpSample_3_layer_call_and_return_conditional_losses_33520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_De2DTrans_3_layer_call_fn_33571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
F__inference_De2DTrans_3_layer_call_and_return_conditional_losses_33561?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
1__inference_Conv2DTrans_recon_layer_call_fn_33620?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
L__inference_Conv2DTrans_recon_layer_call_and_return_conditional_losses_33610?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?B?
#__inference_signature_wrapper_33977Conv2D_1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
L__inference_Conv2DTrans_recon_layer_call_and_return_conditional_losses_33610?PQI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
1__inference_Conv2DTrans_recon_layer_call_fn_33620?PQI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
C__inference_Conv2D_1_layer_call_and_return_conditional_losses_34318p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
(__inference_Conv2D_1_layer_call_fn_34327c9?6
/?,
*?'
inputs???????????
? ""???????????? ?
C__inference_Conv2D_2_layer_call_and_return_conditional_losses_34338p9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????@
? ?
(__inference_Conv2D_2_layer_call_fn_34347c9?6
/?,
*?'
inputs??????????? 
? ""????????????@?
C__inference_Conv2D_3_layer_call_and_return_conditional_losses_34358m()7?4
-?*
(?%
inputs?????????@@@
? ".?+
$?!
0?????????@@?
? ?
(__inference_Conv2D_3_layer_call_fn_34367`()7?4
-?*
(?%
inputs?????????@@@
? "!??????????@@??
F__inference_De2DTrans_2_layer_call_and_return_conditional_losses_33497?@AJ?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_De2DTrans_2_layer_call_fn_33507?@AJ?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
F__inference_De2DTrans_3_layer_call_and_return_conditional_losses_33561?JKI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_De2DTrans_3_layer_call_fn_33571?JKI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
E__inference_DeConv2D_1_layer_call_and_return_conditional_losses_33433?67J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
*__inference_DeConv2D_1_layer_call_fn_33443?67J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
D__inference_MaxPool_1_layer_call_and_return_conditional_losses_33349?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_MaxPool_1_layer_call_fn_33355?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_MaxPool_2_layer_call_and_return_conditional_losses_33361?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_MaxPool_2_layer_call_fn_33367?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_MaxPool_3_layer_call_and_return_conditional_losses_33373?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_MaxPool_3_layer_call_fn_33379?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_UpSample_1_layer_call_and_return_conditional_losses_33392?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_UpSample_1_layer_call_fn_33398?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_UpSample_2_layer_call_and_return_conditional_losses_33456?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_UpSample_2_layer_call_fn_33462?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_UpSample_3_layer_call_and_return_conditional_losses_33520?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_UpSample_3_layer_call_fn_33526?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
 __inference__wrapped_model_33343?()67@AJKPQA?>
7?4
2?/
Conv2D_1_input???????????
? "O?L
J
Conv2DTrans_recon5?2
Conv2DTrans_recon????????????
E__inference_sequential_layer_call_and_return_conditional_losses_33732?()67@AJKPQI?F
??<
2?/
Conv2D_1_input???????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_33777?()67@AJKPQI?F
??<
2?/
Conv2D_1_input???????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_34109?()67@AJKPQA?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_34241?()67@AJKPQA?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
*__inference_sequential_layer_call_fn_33856?()67@AJKPQI?F
??<
2?/
Conv2D_1_input???????????
p

 
? "2?/+????????????????????????????
*__inference_sequential_layer_call_fn_33934?()67@AJKPQI?F
??<
2?/
Conv2D_1_input???????????
p 

 
? "2?/+????????????????????????????
*__inference_sequential_layer_call_fn_34274?()67@AJKPQA?>
7?4
*?'
inputs???????????
p

 
? "2?/+????????????????????????????
*__inference_sequential_layer_call_fn_34307?()67@AJKPQA?>
7?4
*?'
inputs???????????
p 

 
? "2?/+????????????????????????????
#__inference_signature_wrapper_33977?()67@AJKPQS?P
? 
I?F
D
Conv2D_1_input2?/
Conv2D_1_input???????????"O?L
J
Conv2DTrans_recon5?2
Conv2DTrans_recon???????????