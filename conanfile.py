from conans import ConanFile, CMake

class ExampleConan(ConanFile):
    name = "imaging"
    version = "1.0"
    settings = None
    exports = "README.md"
    #conan libs you would like to use
    requires = "lms/2.0@lms/stable","cereal/1.2-0@lms/stable","gtest/1.8.0@lms/stable","Eigen3/3.2.8@bilke/stable"

    def imports(self):
        self.copy("*.so",dst=".")
        self.copy("*.a",dst=".")
        self.copy("*.dylib",dst=".")
        self.copy("bin/*",dst=".")
        self.copy("include/*",dst="")
        #self.copy("Eigen/*",dst="include/")
