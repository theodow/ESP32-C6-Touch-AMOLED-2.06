set(srcs main.c imu_window.c fall_detector.c fall_heuristic.c)
set(priv_requires driver nvs_flash esp_timer)

if(CONFIG_FALL_BACKEND_TFLM)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/model/fall_model_data.cc)
        message(FATAL_ERROR
            "FALL_BACKEND_TFLM is enabled but main/model/fall_model_data.cc is missing. "
            "Run training/train_fall_model.py to generate it (see README.md).")
    endif()
    list(APPEND srcs fall_model_tflm.cc model/fall_model_data.cc)
    list(APPEND priv_requires esp-tflite-micro)
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS . model
    PRIV_REQUIRES ${priv_requires}
)
