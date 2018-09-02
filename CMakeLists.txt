cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(examples)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
add_subdirectory(/home/user/dlib dlib_build)


macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib )
endmacro()

# if an example requires GUI, call this macro to check DLIB_NO_GUI_SUPPORT to include or exclude
macro(add_gui_example name)
   if (DLIB_NO_GUI_SUPPORT)
      message("No GUI support, so we won't build the ${name} example.")
   else()
      add_example(${name})
   endif()
endmacro()


if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()

# csv-text header
include_directories(/home/user/OBJ2IMG/includes/)
#here we apply our macros 
add_example(training_paramters)



