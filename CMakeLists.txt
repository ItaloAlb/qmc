# --- Configuração Mínima e Definição do Projeto ---
# Define a versão mínima do CMake necessária. 3.10 é uma escolha moderna e segura.
cmake_minimum_required(VERSION 3.10)
add_compile_definitions(_USE_MATH_DEFINES)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Define o nome do projeto, sua versão e a linguagem de programação.
project(
    QMC_Project
    VERSION 1.0
    LANGUAGES CXX
)

# --- Configurações do Compilador ---
# Define o padrão do C++ para C++17.
# O CMake se encarregará de adicionar o flag correto (ex: -std=c++17).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependências ---
# Procura pela biblioteca OpenMP, que é necessária para a paralelização.
# Se não for encontrada, o CMake emitirá um erro.
find_package(OpenMP REQUIRED)

# Baixa a biblioteca nlohmann/json para parsing de JSON.
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# --- Definição do Executável e Arquivos-Fonte ---
# Define o nome do arquivo executável que será gerado.
set(EXECUTABLE_NAME qmc)

# Cria uma lista com todos os seus arquivos de implementação (.cpp).
# Se você adicionar um novo arquivo na pasta 'src', lembre-se de adicioná-lo a esta lista.
set(SOURCES
    src/main.cpp
    src/qmc.cpp
    src/vmc.cpp
    src/dmc.cpp
    src/wavefunction.cpp
    src/hamiltonian.cpp
    src/periodic_boundary.cpp
    src/utils.cpp
)

# Adiciona o executável ao projeto, informando quais arquivos-fonte o compõem.
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# --- Configuração de Inclusão e Vinculação (Linking) ---
# Informa ao compilador onde encontrar os arquivos de cabeçalho (.h).
# ${PROJECT_SOURCE_DIR} é uma variável do CMake que aponta para a pasta raiz do projeto (qmc/).
target_include_directories(${EXECUTABLE_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Vincula (linka) o executável com as bibliotecas externas necessárias.
# Aqui, estamos vinculando com o OpenMP. O CMake adiciona automaticamente os flags
# de compilação e de vinculação necessários (ex: -fopenmp).
target_link_libraries(${EXECUTABLE_NAME}
    PUBLIC
        OpenMP::OpenMP_CXX
        nlohmann_json::nlohmann_json
)

# --- Configuração de Saída (Opcional, mas recomendado) ---
# Define que o executável final deve ser colocado em uma pasta 'bin' dentro da pasta 'build'.
# Isso mantém a estrutura de pastas da compilação mais organizada.
set_target_properties(${EXECUTABLE_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --- Mensagem de Finalização ---
# Imprime uma mensagem útil quando o CMake termina de configurar o projeto.
message(STATUS "Configuração concluída. Para compilar, entre na pasta 'build' e execute 'make'.")