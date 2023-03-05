# STARK101-rs 🦀

## Acerca de

Este repositorio está basado en el taller [STARK 101](https://github.com/starkware-industries/stark101), originalmente escrito en Python.

Un tutorial de Rust para un protocolo básico STARK (**S**calable **T**ransparent **AR**gument of **K**nowledge) que demuestra el cálculo de una secuencia de Fibonacci-cuadrado, diseñado para las sesiones de StarkWare y creado por el equipo de [StarkWare](https://starkware.co).

Tenga en cuenta que se supone que el usuario ha revisado y entendido las presentaciones al comienzo de cada parte.

## Configuración

Para seguir este taller necesitas:

- Tener instalado Rust y Jupyter
- Instalar evcxr_jupyter
  > cargo install evcxr_jupyter
- Ejecuta el siguiente comando para registrar el kernel de Rust:
  > evcxr_jupyter --install
- Ejecuta Jupyter
  > jupyter lab

## Antecedentes Matemáticos

Durante el tutorial, generará una prueba STARK para el 1023<sup>rd</sup> elemento de la
secuencia FibonacciSq en un campo finito. En esta sección, explicamos lo que significa esta última oración.

### Campos Finitos

En el tutorial trabajaremos con un campo finito de tamaño primo. Esto significa que tomamos un número primo _p_, y luego trabajamos con enteros en el dominio {0, 1, 2, ..., _p_ - 1}. La idea es que podemos tratar este conjunto de enteros de la misma manera que tratamos los números reales: podemos sumarlos (pero necesitamos tomar el resultado módulo _p_, para que vuelva al conjunto), restarlos, multiplicarlos y dividirlos. Incluso se pueden definir polinomios como _f_ (_x_) = _a_ + _bx_<sup>2</sup> donde los coeficientes _a_,_b_ y la entrada _x_ son todos números en este conjunto finito. Como la adición y la multiplicación se hacen módulo _p_, la salida _f_ (_x_) también estará en el conjunto finito. Una cosa interesante a destacar de los campos finitos, que es diferente de los números reales, es que siempre hay un elemento _g_, llamado generador (de hecho, hay más de uno), para el cual la secuencia 1, _g_, _g_<sup>2</sup>, _g_<sup>3</sup>, _g_<sup>4</sup>, ..., _g_<sup>p-2</sup> (cuya longitud es _p_ - 1) cubre todos los números del conjunto excepto 0 (módulo _p_, por supuesto). Tal secuencia geométrica se llama grupo cíclico. Le proporcionaremos clases de Python que implementan estas cosas para que no tenga que estar familiarizado con cómo se implementan (aunque el algoritmo de división en un campo finito no es tan trivial).

### FibonacciSq

Para el tutorial definimos una secuencia que se asemeja a la conocida secuencia de Fibonacci. En esta secuencia, cualquier elemento es la suma de los cuadrados de los dos elementos anteriores. Por lo tanto, los primeros elementos son:

1, 1, 2, 5, 29, 866, ...

Todos los elementos de la secuencia serán del campo finito (lo que significa que tanto el cuadrado como la adición se calculan módulo p).

### Prueba STARK

Crearemos una prueba para la afirmación "El elemento 1023<sup>rd</sup> de la secuencia FibonacciSq es...". Por "prueba" no nos referimos a una prueba matemática con deducciones lógicas, sino a algunos datos que puedan convencer a quien los lea de que la afirmación es correcta. Para hacerlo más formal, definimos dos entidades: el **Probador** y el **Verificador**. El Probador genera estos datos (proof). El Verificador recibe estos datos y verifica su validez. El requisito es que si la afirmación es falsa, el Probador no podrá generar una prueba válida (incluso si se desvía del protocolo).

STARK es un protocolo específico que describe la estructura de tal prueba y define lo que el Probador y el Verificador tienen que hacer.

### Algunas Otras Cosas que Debes Saber

Le recomendamos que eche un vistazo a nuestras publicaciones matemáticas de [STARK math blog
posts](https://medium.com/starkware/tagged/stark-math) (Arithmetization
[I](https://medium.com/starkware/arithmetization-i-15c046390862) &
[II](https://medium.com/starkware/arithmetization-ii-403c3b3f4355) specifically). No es necesario que las lea a fondo antes de seguir este tutorial, pero puede darle un mejor contexto sobre las cosas para las que puede crear pruebas y cómo se ve una prueba de STARK. Definitivamente debería leerlas después de completar este tutorial por completo.

### División de Polinomios

Para cada dos polinomios _f_ ( _x_ ) y _g_ ( _x_ ), existen dos polinomios _q_ ( _x_ ) y
_r_ ( _x_) llamados el cociente y el resto de la división  _f_ ( _x_ ) por _g_ ( _x_ ). Satisfacen _f_ ( _x_ ) = _g_ ( _x_ ) \* _q_ ( _x_ ) + _r_ ( _x_ ) y el grado de  _r_ ( _x_ ) es menor que el grado de _g_ ( _x_ ).

Por ejemplo, si _f_ ( _x_ ) = _x_<sup>3</sup> + _x_ + 1 and _g_ ( _x_ ) = _x_<sup>2</sup> + 1 entonces
_q_ ( _x_ ) = _x_ y _r_ ( _x_ ) = 1. De hecho, _x_<sup>3</sup> + _x_ + 1 = ( _x_<sup>2</sup> + 1 )
\* _x_ + 1.

### Raíces de Polinomios

Cuando un polinomio satisface _f_ (_a_) = 0 para algún valor específico _a_ (decimos que _a_ es una raíz de _f_), entonces no tenemos un residuo (_r_ ( _x_ ) = 0) al dividirlo por  (_x_ - _a_), por lo que podemos escribir _f_ (_x_) = ((_x_ - _a_) \* _q_ ( _x_ ), y deg(_q_) = deg(_f_) - 1. Un hecho similar es cierto para _k_ raíces. Es decir, si _a_<sub>i</sub> es una raíz de _f_ para todos los _i_ = 1, 2, …, _k_, entonces existe un polinomio _q_ de grado deg(_f_) - _k_ para el cual  _f_ ( _x_ ) = ( _x_ - _a_<sub>1</sub> )( _x_ -
_a_<sub>2</sub> ) … ( _x_ - _a_<sub>_k_</sub> ) \* _q_ ( _x_ ) .

### ¿Quieres saber más?

1. Nigel Smart [“Cryptography Made Simple”](https://www.cs.umd.edu/~waa/414-F11/IntroToCrypto.pdf): 
   – Capítulo 1.1: Aritmética modular.
2. [“Computational Complexity: A Modern
   Approach”](http://theory.cs.princeton.edu/complexity/book.pdf) –  Apéndice: Antecedentes matemáticos, secciones A.4 (Campos finitos y grupos) y A.6 (Polinomios).
