/* BSD 3-Clause License

Copyright (c) 2021, Event Driven Perception for Robotics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#include <hpe-core/utility.h>
#include <hpe-core/representations.h>
#include <hpe-core/fusion.h>
#include <hpe-core/motion_estimation.h>

bool test_representations()
{

    cv::namedWindow("EROS");
    cv::namedWindow("BINARY");
    cv::namedWindow("SAE");

    hpecore::EROS eros;
    hpecore::BIN binary;
    hpecore::SAE sae;

    eros.init(640, 480, 7, 0.3);
    binary.init(640, 480, 0);
    sae.init(640, 480, 0);

    bool exit = false;

    double t = 0;
    while (!exit) {
        for (int y = 0; y < 480; y++) {
            for (int x = 0; x < 640; x++) {
                eros.update(x, y);
                binary.update(x, y);
                sae.update(x, y, t);
            }
            if(++t > 1000.0) t = 0;
            cv::imshow("EROS", eros.getSurface() / 255.0);
            cv::imshow("BINARY", binary.getSurface() / 255.0);
            cv::imshow("SAE", sae.getSurface() / 1000.0);
            char c = cv::waitKey(1);
            if(c == '\e') exit = true;
        }
        binary.getSurface().setTo(0.0);
    }

    cv::destroyAllWindows();

    return true;
}

int main(int argc, char* argv[])
{
    
    test_representations();




    return 0;

}