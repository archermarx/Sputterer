#version 330 core
out vec4 fragColor;

in vec3 normal;
in vec3 fragPos;

const float ambientStrength = 0.1;
const float specularStrength = 0.1;
const int shininess = 8;
const vec3 lightColor = vec3(1.0, 1.0, 1.0);
const vec3 lightPos = vec3(10.0, 40.0, 8.0);
const float lightPower = length(lightPos) * length(lightPos);
const float screenGamma = 2.2;

uniform vec3 viewPos;
uniform vec3 objectColor;

void main() {
    vec3 norm = normalize(normal);
    vec3 lightDir = lightPos - fragPos;
    float distance = length(lightDir) * length(lightDir);
    lightDir = normalize(lightDir);
    vec3 viewDir = normalize(viewPos - fragPos);

    vec3 ambient = ambientStrength * lightColor;

    float lambertian = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = lambertian * lightColor * lightPower / distance;

    float spec = 0.0;

    if(lambertian > 0){
        vec3 reflectDir = normalize(lightDir + viewDir);
        float specAngle = max(dot(reflectDir, viewDir), 0.0);
        spec = pow(specAngle, shininess);
    }

    vec3 specular = specularStrength * spec * lightColor * lightPower / distance;

    vec3 resultColor = (ambient + diffuse + specular) * objectColor;
    vec3 resultColorCorrected = pow(resultColor, vec3(1.0 / screenGamma));
    fragColor = vec4(resultColorCorrected, 1.0);
}